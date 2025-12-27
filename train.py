import os
import torch
from unsloth import FastLanguageModel
from transformers import TrainingArguments, Trainer
from config import ModelConfig, TrainConfig
from dataset_prep import prepare_dataset, AudioTokenizer
from dataclasses import asdict

def train(metadata_path: str, overrides: dict = None):
    """
    Main training entry point.
    
    Args:
        metadata_path (str): Path to the metadata.csv file.
        overrides (dict): Optional dictionary to override TrainConfig values (e.g., max_steps, learning_rate).
    """
    # 1. Load Configurations
    model_config = ModelConfig()
    train_config = TrainConfig()
    
    # Apply overrides to train_config
    if overrides:
        for key, value in overrides.items():
            if hasattr(train_config, key):
                setattr(train_config, key, value)
            else:
                print(f"Warning: Ignoring unknown config override '{key}'")

    print(f"Starting training with config: {train_config}")

    # 2. Load Model & Tokenizer (Unsloth)
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_config.model_name,
        max_seq_length=model_config.max_seq_length,
        dtype=None,  # Auto detection
        load_in_4bit=model_config.load_in_4bit,
    )

    # 3. Add LoRA Adapters
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=model_config.r,
        target_modules=model_config.target_modules,
        lora_alpha=model_config.lora_alpha,
        lora_dropout=model_config.lora_dropout,
        bias=model_config.bias,
        use_gradient_checkpointing=model_config.use_gradient_checkpointing,
        random_state=model_config.random_state,
        use_rslora=False,
        loftq_config=None,
    )

    # 4. Prepare Dataset
    print(f"Preparing dataset from {metadata_path}...")
    audio_tokenizer = AudioTokenizer(device="cuda" if torch.cuda.is_available() else "cpu")
    
    # dataset_prep.prepare_dataset expects: csv_path, tokenizer, audio_tokenizer
    dataset = prepare_dataset(metadata_path, tokenizer, audio_tokenizer)
    
    # 5. Setup Trainer
    print("Setting up Trainer...")
    training_args = TrainingArguments(
        output_dir=train_config.output_dir,
        per_device_train_batch_size=train_config.per_device_train_batch_size,
        gradient_accumulation_steps=train_config.gradient_accumulation_steps,
        warmup_steps=train_config.warmup_steps,
        max_steps=train_config.max_steps,
        learning_rate=train_config.learning_rate,
        logging_steps=50,
        optim=train_config.optim,
        weight_decay=train_config.weight_decay,
        lr_scheduler_type="linear",
        seed=train_config.seed,
        report_to="none", # Change to "wandb" or "tensorboard" if needed
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
    )

    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
    )

    # 6. Train
    print("Starting training loop...")
    trainer_stats = trainer.train()
    
    # 7. Save Model
    print(f"Saving model to {train_config.output_dir}...")
    model.save_pretrained(train_config.output_dir+"lora_model")
    tokenizer.save_pretrained(train_config.output_dir+"lora_model")
    
    return f"Training complete. Model saved to {train_config.output_dir}"

if __name__ == "__main__":
    # Test run (requires a valid dataset/metadata.csv to actually work)
    # train("dataset/metadata.csv")
    pass
