import sys
from transformers import HfArgumentParser
from src.config.train_config import ModelArguments, DataArguments, WhisperTrainingArguments
from src.models.whisper_model import load_whisper_model
from src.data.dataset import ChineseTaiwaneseDataset
from src.data.data_collator import WhisperDataCollator
from src.trainers.whisper_trainer import get_trainer

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, WhisperTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=sys.argv[1])
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set fp16 to False if not available
    training_args.fp16 = training_args.fp16 and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7

    model, processor = load_whisper_model(
        model_args.model_name_or_path, 
        use_peft=model_args.use_peft, 
        peft_config={
            "task_type": "SPEECH_RECOGNITION",
            "r": model_args.lora_r,
            "lora_alpha": model_args.lora_alpha,
            "lora_dropout": model_args.lora_dropout,
        } if model_args.use_peft else None
    )

    train_dataset = ChineseTaiwaneseDataset(data_args.dataset_name, "train", processor, 
                                            text_column=data_args.text_column,
                                            audio_column=data_args.audio_column,
                                            max_samples=data_args.max_train_samples,
                                            language=model_args.language)
    eval_dataset = ChineseTaiwaneseDataset(data_args.dataset_name, "validation", processor, 
                                           text_column=data_args.text_column,
                                           audio_column=data_args.audio_column,
                                           max_samples=data_args.max_eval_samples,
                                           language=model_args.language)

    data_collator = WhisperDataCollator(processor=processor)

    trainer = get_trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processor=processor
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    main()