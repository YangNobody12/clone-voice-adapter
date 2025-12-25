import torch
import torchaudio.transforms as T
from snac import SNAC
from datasets import Dataset, load_dataset
import pandas as pd
import os

class AudioTokenizer:
    def __init__(self, device="cuda"):
        self.device = device
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device)
        self.sample_rate = 24000
    
    def tokenise_audio(self, audio_array, orig_sr):
        waveform = torch.from_numpy(audio_array).unsqueeze(0).to(dtype=torch.float32)
        
        if orig_sr != self.sample_rate:
            resample_transform = T.Resample(orig_freq=orig_sr, new_freq=self.sample_rate)
            waveform = resample_transform(waveform)
            
        waveform = waveform.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            codes = self.snac_model.encode(waveform)

        all_codes = []
        # Interleave codes logic from original script
        for i in range(codes[0].shape[1]):
            all_codes.append(codes[0][0][i].item()+128266)
            all_codes.append(codes[1][0][2*i].item()+128266+4096)
            all_codes.append(codes[2][0][4*i].item()+128266+(2*4096))
            all_codes.append(codes[2][0][(4*i)+1].item()+128266+(3*4096))
            all_codes.append(codes[1][0][(2*i)+1].item()+128266+(4*4096))
            all_codes.append(codes[2][0][(4*i)+2].item()+128266+(5*4096))
            all_codes.append(codes[2][0][(4*i)+3].item()+128266+(6*4096))
            
        return all_codes

    def redistribute_codes(self, code_list):
        """Reconstruct codes for SNAC decoding"""
        layer_1, layer_2, layer_3 = [], [], []
        for i in range((len(code_list)+1)//7):
            layer_1.append(code_list[7*i])
            layer_2.append(code_list[7*i+1]-4096)
            layer_3.append(code_list[7*i+2]-(2*4096))
            layer_3.append(code_list[7*i+3]-(3*4096))
            layer_2.append(code_list[7*i+4]-(4*4096))
            layer_3.append(code_list[7*i+5]-(5*4096))
            layer_3.append(code_list[7*i+6]-(6*4096))
            
        codes = [
            torch.tensor(layer_1).unsqueeze(0).to(self.device),
            torch.tensor(layer_2).unsqueeze(0).to(self.device),
            torch.tensor(layer_3).unsqueeze(0).to(self.device)
        ]
        return codes

    def decode_codes(self, codes):
        return self.snac_model.decode(codes)


def remove_duplicate_frames(example):
    vals = example["codes_list"]
    if len(vals) % 7 != 0:
        return example # or raise Error

    result = vals[:7]
    for i in range(7, len(vals), 7):
        current_first = vals[i]
        previous_first = result[-7]

        if current_first != previous_first:
            result.extend(vals[i:i+7])
    
    example["codes_list"] = result
    return example

def prepare_dataset(csv_path, tokenizer, audio_tokenizer: AudioTokenizer):
    """
    Loads dataset from CSV (audio, text, source) or LJSpeech format (path|text) 
    and prepares it for training
    """
    # Define tokens
    tokeniser_length = 128256
    start_of_human = tokeniser_length + 3
    end_of_human = tokeniser_length + 4
    start_of_ai = tokeniser_length + 5
    end_of_ai =  tokeniser_length + 6
    start_of_speech = tokeniser_length + 1
    end_of_speech = tokeniser_length + 2
    end_of_text = 128009

    # Detect format and load accordingly
    with open(csv_path, 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
    
    print(f"[Dataset] First line: {first_line[:80]}...")
    print(f"[Dataset] Contains '|': {'|' in first_line}, Contains ',': {',' in first_line}")
    
    if '|' in first_line and ',' not in first_line:
        # LJSpeech format: path|text
        print("[Dataset] Detected LJSpeech format (path|text)")
        data = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and '|' in line:
                    parts = line.split('|', 1)
                    if len(parts) == 2:
                        data.append({
                            'audio': parts[0],
                            'text': parts[1],
                            'source': 'speaker'
                        })
        df = pd.DataFrame(data)
        print(f"[Dataset] Loaded {len(df)} rows from LJSpeech format")
        print(f"[Dataset] Columns: {list(df.columns)}")
        print(f"[Dataset] Sample: {df.head(1).to_dict()}")
    else:
        # CSV format with header
        print("[Dataset] Detected CSV format (audio,text,source)")
        df = pd.read_csv(csv_path)
        if 'source' not in df.columns:
            df['source'] = 'speaker'
    
    dataset = Dataset.from_pandas(df)
    
    # 1. Tokenise Audio
    def add_codes(example):
        try:
            # Assuming audio path is relative to csv location or absolute
            # We need to load audio here. 
            # If dataset was loaded via load_dataset("audiofolder"), it would handle this.
            # But we are doing manual CSV load.
            audio_path = example["audio"]
            # Fix path if needed (relative to csv)
            if not os.path.exists(audio_path) and os.path.exists(os.path.join(os.path.dirname(csv_path), audio_path)):
                audio_path = os.path.join(os.path.dirname(csv_path), audio_path)
                
            import soundfile as sf
            wav, sr = sf.read(audio_path)
            codes = audio_tokenizer.tokenise_audio(wav, sr)
            example["codes_list"] = codes
        except Exception as e:
            print(f"Error processing {example}: {e}")
            example["codes_list"] = None
        return example

    dataset = dataset.map(add_codes)
    dataset = dataset.filter(lambda x: x["codes_list"] is not None)
    dataset = dataset.filter(lambda x: len(x["codes_list"]) > 0)
    
    # 2. Remove duplicates
    dataset = dataset.map(remove_duplicate_frames)
    
    # 3. Create Inputs
    def create_input_ids(example):
        text_prompt = f"{example['source']}: {example['text']}" if "source" in example else example["text"]
        
        text_ids = tokenizer.encode(text_prompt, add_special_tokens=True)
        text_ids.append(end_of_text)
        
        input_ids = (
            [start_of_human]
            + text_ids
            + [end_of_human]
            + [start_of_ai]
            + [start_of_speech]
            + example["codes_list"]
            + [end_of_speech]
            + [end_of_ai]
        )
        
        example["input_ids"] = input_ids
        example["labels"] = input_ids
        example["attention_mask"] = [1] * len(input_ids)
        return example

    dataset = dataset.map(create_input_ids)
    
    return dataset
