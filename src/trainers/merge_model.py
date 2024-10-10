import argparse
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, AutoTokenizer
# from transformers import (
#     WhisperForConditionalGeneration,
#     WhisperProcessor,
#     WhisperTokenizer,
# )
from peft import PeftModel, PeftConfig
from src.utils.logging import logger


def merge_and_save_whisper_model(base_model_name, peft_model_path, output_dir):
    logger.info("[1/7] Loading PeftConfig")
    config = PeftConfig.from_pretrained(peft_model_path)

    logger.info("[2/7] Loading base Whisper model")
    base_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    logger.info("[3/7] Loading LoRA adapter")
    model = PeftModel.from_pretrained(base_model, peft_model_path, config=config)

    logger.info("[4/7] Merging base model and adapter")
    model = model.merge_and_unload()

    logger.info("[5/7] Saving merged model")
    model.save_pretrained(output_dir)

    logger.info("[6/7] Saving WhisperProcessor")
    processor = AutoProcessor.from_pretrained(base_model_name)
    processor.save_pretrained(output_dir)

    logger.info("[7/7] Saving WhisperTokenizer")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.save_pretrained(output_dir)

    logger.info(f"Merged model, processor, and tokenizer saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge Whisper LoRA and base model and save the result"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path or name of the base Whisper model",
    )
    parser.add_argument(
        "--peft_model", type=str, required=True, help="Path to the PEFT (LoRA) model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the merged model",
    )

    args = parser.parse_args()

    merge_and_save_whisper_model(args.base_model, args.peft_model, args.output_dir)


if __name__ == "__main__":
    main()
