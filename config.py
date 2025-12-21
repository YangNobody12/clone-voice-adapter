from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class ModelConfig:
    model_name: str = "unsloth/orpheus-3b-0.1-ft"
    max_seq_length: int = 2048
    load_in_4bit: bool = False
    
    # LoRA Config
    r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    bias: str = "none"
    use_gradient_checkpointing: str = "unsloth"
    random_state: int = 3407
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])

@dataclass
class TrainConfig:
    output_dir: str = "outputs"
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 5
    max_steps: int = 60
    learning_rate: float = 2e-4
    optim: str = "adamw_8bit"
    weight_decay: float = 0.001
    seed: int = 3407
    
    # Dataset
    dataset_path: str = "dataset" # placeholder or path to processed dataset
