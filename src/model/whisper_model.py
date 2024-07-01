from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import get_peft_model, LoraConfig, TaskType

def load_whisper_model(model_name_or_path: str, use_peft: bool = False, peft_config: dict = None):
    model = WhisperForConditionalGeneration.from_pretrained(model_name_or_path)
    processor = WhisperProcessor.from_pretrained(model_name_or_path)

    if use_peft:
        if peft_config is None:
            peft_config = {
                "task_type": TaskType.SPEECH_RECOGNITION,
                "r": 8,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
            }
        lora_config = LoraConfig(**peft_config)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, processor