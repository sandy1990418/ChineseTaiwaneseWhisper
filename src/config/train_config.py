from dataclasses import dataclass, field
from typing import Optional, List
from transformers import Seq2SeqTrainingArguments
import torch
import os


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="openai/whisper-small",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    language: str = field(
        default="chinese",
        metadata={"help": "Language code for the model (e.g., zh-TW for Traditional Chinese)"}
    )
    use_peft: bool = field(
        default=False,
        metadata={"help": "Whether to use PEFT for fine-tuning"}
    )
    peft_method: str = field(
        default="lora",
        metadata={"help": "PEFT method to use: lora, ia3, adaption_prompt, prefix_tuning, p_tuning"}
    )
    lora_r: int = field(
        default=8,
        metadata={"help": "LoRA attention dimension"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "LoRA dropout"}
    )


@dataclass
class DataArguments:
    dataset_name: str = field(
        default="mozilla-foundation/common_voice_11_0",
        metadata={"help": "The name of the dataset to use (via the datasets library)"}
    )
    dataset_config_name: Optional[str] = field(
        default="zh-TW", 
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)"}
    )
    text_column: str = field(
        default="sentence",
        metadata={"help": "The name of the column in the datasets containing the full texts."},
    )
    audio_column: str = field(
        default="audio",
        metadata={"help": "The name of the column in the datasets containing the audio files"},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of \
                  training examples to this value if set."}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of\
                   evaluation examples to this value if set."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=1,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    youtube_data_dir: str = field(
        default="./youtube_data",
        metadata={"help": "Directory containing YouTube audio and subtitle data"}
    )
    max_input_length: int = field(
        default=30,
        metadata={"help": "Maximum input length in seconds for audio clips"}
    )
    use_timestamps: bool = field(
        default=False,
        metadata={"help": "Use timestamp or not."}
    )


@dataclass
class WhisperTrainingArguments(Seq2SeqTrainingArguments):
    output_dir: str = field(
        default="./whisper-finetuned",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    per_device_train_batch_size: int = field(
        default=16, 
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    num_train_epochs: float = field(
        default=3.0, 
        metadata={"help": "Total number of training epochs to perform."}
    )
    warmup_steps: int = field(
        default=500, 
        metadata={"help": "Linear warmup over warmup_steps."}
    )
    learning_rate: float = field(
        default=3e-4, 
        metadata={"help": "The initial learning rate for AdamW."}
    )
    fp16: bool = field(
        default=True,
        metadata={"help": "Whether to use 16-bit (mixed) precision instead of 32-bit"}
    )
    evaluation_strategy: str = field(
        default="steps",
        metadata={"help": "The evaluation strategy to use."}
    )
    save_steps: int = field(
        default=1000,
        metadata={"help": "Save checkpoint every X updates steps."}
    )
    eval_steps: int = field(
        default=1000,
        metadata={"help": "Run an evaluation every X steps."}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "Log every X updates steps."}
    )
    save_total_limit: Optional[int] = field(
        default=2,
        metadata={"help": "Limit the total amount of checkpoints."}
    )
    metric_for_best_model: str = field(
        default="wer",
        metadata={"help": "The metric to use to compare two different models."}
    )
    greater_is_better: bool = field(
        default=False,
        metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."}
    )
    load_best_model_at_end: bool = field(
        default=True,
        metadata={"help": "Whether or not to load the best model found during training at the end of training."}
    )
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."}
    )


@dataclass
class CrawlerArgs:
    # List of YouTube playlist URLs to crawl
    playlist_urls: List[str] = field(
        default_factory=list,
        metadata={"help": "YouTube playlist URLs to crawl"}
    )

    # Directory to save audio files and dataset
    output_dir: str = field(
        default="./output",
        metadata={"help": "Directory to save audio files and dataset"}
    )

    # Name of the output dataset file
    dataset_name: str = field(
        default="youtube_dataset",
        metadata={"help": "Name of the output dataset file"}
    )

    # Path to FFmpeg executable (optional)
    ffmpeg_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to FFmpeg executable"}
    )

    # Prefix for audio and subtitle files
    file_prefix: str = field(
        default="youtube",
        metadata={"help": "Prefix for audio and subtitle files"}
    )


@dataclass
class InferenceArguments:
    model_path: str = field(
        metadata={"help": "Path to the ASR model"}
    )
    audio_files: List[str] = field(
        default_factory=list,
        metadata={"help": "Path to audio file(s)"}
    )
    mode: str = field(
        default="batch",
        metadata={"help": "Inference mode", "choices": ["batch", "stream"]}
    )
    use_timestamps: bool = field(
        default=False,
        metadata={"help": "Include timestamps in transcription"}
    )
    use_peft: bool = field(
        default=False,
        metadata={"help": "Use PEFT model"}
    )
    language: str = field(
        default="chinese",
        metadata={"help": "Language of the audio (e.g., 'chinese', 'taiwanese')"}
    )
    device: str = field(
        default=None,
        metadata={"help": "Device to use for inference"}
    )
    output_dir: str = field(
        default="./output",
        metadata={"help": "Directory to save output files"}
    )
    file_name: str = field(
        default="translation_result.json",
        metadata={"help": "Directory to save output files"}
    )
    
    def __post_init__(self):
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        os.makedirs(self.output_dir, exist_ok=True)


@dataclass
class GradioArguments:
    cache_dir: str = field(
        default="asr_transcription_streaming_cache",
        metadata={"help": "Path to store ASR result"}
    )
    cache_streaming_filename: str = field(
        default="asr_log_streaming.json",
        metadata={"help": "File name to store ASR result"}
    )
    cache_batch_filename: str = field(
        default="asr_log_batch.json",
        metadata={"help": "File name to store ASR result"}
    )
    language: str = field(
        default="chinese",
        metadata={"help": "Language code for the model (e.g., zh-TW for Traditional Chinese)"}
    )