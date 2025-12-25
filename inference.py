import torch
from unsloth import FastLanguageModel
from snac import SNAC
import soundfile as sf
import os
import numpy as np
import gc

def cleanup_memory():
    """
    Clean up memory before inference.
    Call this after training to free GPU memory.
    """
    # Try to delete common training variables if they exist in global scope
    for var_name in ['model', 'optimizer', 'data', 'trainer', 'dataset']:
        if var_name in globals():
            del globals()[var_name]
    
    # Run garbage collection
    gc.collect()
    
    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print("Memory cleaned up successfully!")

class InferenceEngine:
    def __init__(self, model_path, device="cuda"):
        self.device = device
        self.model_path = model_path
        print(f"Loading model from {model_path}...")

        # Load LLM
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=False,
        )
        FastLanguageModel.for_inference(self.model)
        
        # Load Audio Decoder (SNAC)
        print("Loading SNAC model...")
        self.snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to("cpu") # Keep cpu for decoding as in original script? Or move to cuda if needed. Script said "Moving snac_model cuda to cpu"
        
        # Define Special Tokens
        self.tokeniser_length = 128256
        self.start_of_human = 128259 # tokeniser_length + 3
        self.end_of_human = 128260 # tokeniser_length + 4 ?? Wait, let's use exact numbers from original script to be safe
        
        # model.py:
        # start_of_human = tokeniser_length + 3 => 128259
        # end_of_human = tokeniser_length + 4 => 128260
        # But in lines 212-213 of model.py:
        # start_token = 128259
        # end_tokens = [128009, 128260] (End of text, End of human)
        
        self.start_token_tensor = torch.tensor([[128259]], dtype=torch.int64)
        self.end_tokens_tensor = torch.tensor([[128009, 128260]], dtype=torch.int64) 
        self.pad_token_id = 128263

    def generate(self, text, output_file="generated_audio.wav"):
        """
        Generates audio from text and saves to output_file.
        """
        print(f"Generating audio for: '{text}'")
        
        text = f"1: {text}"

        # 1. Prepare Input
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids
        
        # Formulate prompt: <SOH> <Text> <EOT> <EOH>
        # Note: model.py logic: torch.cat([start_token, input_ids, end_tokens], dim=1)
        modified_input_ids = torch.cat([self.start_token_tensor, input_ids, self.end_tokens_tensor], dim=1)
        
        # Padding (simplified from batch logic since we do single item)
        # But model.py does padding to left?
        # padded_tensor = torch.cat([torch.full((1, padding), 128263, ...), modified_input_ids], dim=1)
        # For single item, we might not need padding if we just pass attention mask of 1s.
        
        input_ids_cuda = modified_input_ids.to(self.device)
        attention_mask = torch.ones((1, input_ids_cuda.shape[1]), dtype=torch.int64).to(self.device)
        
        # 2. Generate Tokens
        generated_ids = self.model.generate(
            input_ids=input_ids_cuda,
            attention_mask=attention_mask,
            max_new_tokens=1200,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1,
            num_return_sequences=1,
            eos_token_id=128258, # end_of_speech?
            use_cache=True
        )
        
        # 3. Extract Audio Codes
        # Logic from model.py: find token 128257 (start of speech?) and take everything after?
        # model.py: token_to_find = 128257
        # cropped_tensor = generated_ids[:, last_occurrence_idx+1:]
        
        token_to_find = 128257
        token_to_remove = 128258 # End of speech
        
        token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)
        if len(token_indices[1]) > 0:
            last_idx = token_indices[1][-1].item()
            cropped_tensor = generated_ids[:, last_idx+1:]
        else:
            cropped_tensor = generated_ids
            
        # Remove EOS tokens if present in the sequence (logic from model.py)
        # processed_rows = row[row != 128258]
        
        audio_codes = []
        cropped_list = cropped_tensor[0].tolist() # assuming batch 1
        for t in cropped_list:
            if t != token_to_remove:
                audio_codes.append(t)
                
        # 4. Convert Tokens to SNAC Codes
        # Logic: (t - 128266)
        # Truncate to multiple of 7
        
        new_length = (len(audio_codes) // 7) * 7
        trimmed_codes = audio_codes[:new_length]
        final_codes = [t - 128266 for t in trimmed_codes]
        
        if not final_codes:
            print("Warning: No audio codes generated.")
            return None
            
        # 5. Decode Audio
        audio_tensor = self.redistribute_codes(final_codes)
        
        # 6. Save
        audio_np = audio_tensor.detach().squeeze().cpu().numpy()
        sf.write(output_file, audio_np, 24000)
        
        return output_file

    def redistribute_codes(self, code_list):
        """Reconstruct codes for SNAC decoding"""
        layer_1 = []
        layer_2 = []
        layer_3 = []
        for i in range((len(code_list)+1)//7):
            layer_1.append(code_list[7*i])
            layer_2.append(code_list[7*i+1]-4096)
            layer_3.append(code_list[7*i+2]-(2*4096))
            layer_3.append(code_list[7*i+3]-(3*4096))
            layer_2.append(code_list[7*i+4]-(4*4096))
            layer_3.append(code_list[7*i+5]-(5*4096))
            layer_3.append(code_list[7*i+6]-(6*4096))
            
        codes = [
            torch.tensor(layer_1).unsqueeze(0), #.to(self.device), # model.py had these on CPU before decode?
            torch.tensor(layer_2).unsqueeze(0),
            torch.tensor(layer_3).unsqueeze(0)
        ]
        
        # Note: model.py: "Moving snac_model cuda to cpu"
        # So decoding happens on CPU. Codes should be on CPU.
        
        audio_hat = self.snac_model.decode(codes)
        return audio_hat
