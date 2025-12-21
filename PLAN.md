this my want
1 use model unsloth/orpheus-3b-0.1-ft

2 use library unsloth

3 esy fine tune lora

4 esy use on colab

5 esy use for inference



best config for fine tune
model = FastLanguageModel.get_peft_model(
    model,
    r = 128,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 128, 
    lora_dropout = 0.05,     
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
)


