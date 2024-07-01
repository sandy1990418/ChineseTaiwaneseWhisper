from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import get_peft_model, LoraConfig, TaskType

def load_whisper_model(model_name_or_path: str, use_peft: bool = False, peft_config: dict = None, language: str = "chinese"):
    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)
    processor = WhisperProcessor.from_pretrained(model_name_or_path)

    # Set the language token
    processor.tokenizer.set_prefix_tokens(language=language, task="transcribe")

    if use_peft:
        if peft_config is None:
            peft_config = {
                "task_type": TaskType.SEQ_2_SEQ_LM,
                "r": 8,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
            }
            
        target_modules = []
        for id, (name, param) in enumerate(model.named_modules()):
            if 'model.decoder' in name and ('q_proj' in name or 'v_proj' in name):
                target_modules.append(name)
        peft_config.update({"target_modules": target_modules})

        lora_config = LoraConfig(**peft_config)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, processor