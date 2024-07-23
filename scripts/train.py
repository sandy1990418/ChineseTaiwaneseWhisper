import sys
from transformers import HfArgumentParser
from src.config import ModelArguments, DataArguments, WhisperProcessorConfig, WhisperTrainingArguments
from src.model.whisper_model import load_whisper_model
from src.data.dataset import ChineseTaiwaneseDataset  
from src.data.data_collator import WhisperDataCollator
from src.trainers.whisper_trainer import get_trainer
# from peft import LoraConfig, TaskType


def main():
    parser = HfArgumentParser((ModelArguments, 
                               DataArguments, 
                               WhisperTrainingArguments, 
                               WhisperProcessorConfig))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args, procrssor_args = parser.parse_json_file(json_file=sys.argv[1])
    else:
        model_args, data_args, training_args, procrssor_args = parser.parse_args_into_dataclasses()
    # Configure LoRA if specified
    peft_config = None
    if model_args.use_peft:
        if model_args.peft_method.lower() == 'lora':
            peft_config = {
                "task_type": None,
                "r": model_args.lora_r,
                "lora_alpha": model_args.lora_alpha,
                "lora_dropout": model_args.lora_dropout,
            }
        else:
            raise ValueError(f"Unsupported PEFT method: {model_args.peft_method}")
        
    model, processor = load_whisper_model(
        model_args.model_name_or_path, 
        use_peft=model_args.use_peft,
        peft_config=peft_config,
        language=model_args.language
    )

    # processor.tokenizer.model_max_length = model.config.max_length
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    train_dataset, eval_dataset = ChineseTaiwaneseDataset.create_train_and_test_datasets(
        data_args, 
        processor, 
    )

    data_collator = WhisperDataCollator(
        processor=processor,
    )
    trainer = get_trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processor=processor,
        resume_from_checkpoint=True
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()