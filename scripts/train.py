import sys
from transformers import HfArgumentParser, Seq2SeqTrainingArguments
from src.config.train_config import ModelArguments, DataArguments
from src.model.whisper_model import load_whisper_model
from src.data.dataset import ChineseTaiwaneseDataset  # ChineseTaiwaneseDataset
from src.data.data_collator import WhisperDataCollator
from src.trainers.whisper_trainer import get_trainer
# from peft import LoraConfig, TaskType


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=sys.argv[1])
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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

    train_dataset = ChineseTaiwaneseDataset(data_args.dataset_name, "train", processor, 
                                            text_column=data_args.text_column,
                                            audio_column=data_args.audio_column,
                                            max_samples=data_args.max_train_samples,
                                            dataset_config_name=data_args.dataset_config_name)
    
    eval_dataset = ChineseTaiwaneseDataset(data_args.dataset_name, "validation", processor, 
                                           text_column=data_args.text_column,
                                           audio_column=data_args.audio_column,
                                           max_samples=data_args.max_eval_samples,
                                           dataset_config_name=data_args.dataset_config_name)

    # datasets = create_dataset(
    #     dataset_name=data_args.dataset_name,
    #     processor=processor,
    #     text_column=data_args.text_column,
    #     audio_column=data_args.audio_column,
    #     max_train_samples=data_args.max_train_samples,
    #     max_eval_samples=data_args.max_eval_samples,
    #     youtube_dir=data_args.youtube_data_dir
    # )

    data_collator = WhisperDataCollator(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

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