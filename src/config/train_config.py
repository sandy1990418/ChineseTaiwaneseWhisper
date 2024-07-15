from dataclasses import dataclass, field
from typing import Optional, List, Union
from transformers import Seq2SeqTrainingArguments


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
    dataset_name: Union[str, List[str]] = field(
        default_factory=lambda: ["mozilla-foundation/common_voice_11_0"],
        metadata={"help": "The name of the dataset to use (via the datasets library)"}
    )
    dataset_config_names: Union[str, List[str]] = field(
        default_factory=lambda: None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)"}
    )
    eval_dataset_name: str = field(
        default="mozilla-foundation/common_voice_11_0",
        metadata={"help": "The name of the dataset to use (via the datasets library)"}
    )
    eval_dataset_config_names: str = field(
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
class WhisperProcessorConfig:
    # Feature Extractor Arguments
    feature_size: int = field(
        default=80,
        metadata={"help": "The feature dimension of the extracted features."}
    )
    sampling_rate: int = field(
        default=16000,
        metadata={"help": "The sampling rate at which the audio files should be digitalized expressed in Hertz (Hz)."}
    )
    padding_value: float = field(
        default=0.0,
        metadata={"help": "The value that is used to fill the padding values."}
    )
    do_normalize: bool = field(
        default=True,
        metadata={"help": "Whether to zero-mean and unit-variance normalize the input."}
    )
    return_attention_mask: bool = field(
        default=False,
        metadata={"help": "Whether to return an attention mask."}
    )
    task: str = field(
        default="transcribe",
        metadata={"help": "The task token to use at the start of transcription."}
    )
    predict_timestamps: bool = field(
        default=False,
        metadata={"help": "Whether to predict timestamps."}
    )
    return_timestamps: bool = field(
        default=False,
        metadata={"help": "Whether to return timestamps in the decoded output."}
    )
    model_max_length: int = field(
        default=448,
        metadata={"help": "The maximum length of the model inputs."}
    )
    padding: Union[bool, str] = field(
        default=True,
        metadata={"help": "Padding strategy. Can be bool or 'max_length'."}
    )
    truncation: bool = field(
        default=True,
        metadata={"help": "Whether to truncate sequences longer than model_max_length."}
    )

    # Additional Processing Arguments
    chunk_length_s: Optional[float] = field(
        default=30.0,
        metadata={"help": "The length of audio chunks to process in seconds."}
    )
    stride_length_s: Optional[float] = field(
        default=None,
        metadata={"help": "The length of stride between audio chunks in seconds."}
    )
    ignore_warning: bool = field(
        default=False,
        metadata={"help": "Whether to ignore the warning raised when the audio is too short."}
    )
    
    # Decoding Arguments
    skip_special_tokens: bool = field(
        default=True,
        metadata={"help": "Whether to remove special tokens in the decoding."}
    )
    clean_up_tokenization_spaces: bool = field(
        default=True,
        metadata={"help": "Whether to clean up the tokenization spaces."}
    )

    # Additional Configuration
    forced_decoder_ids: Optional[List[List[int]]] = field(
        default=None,
        metadata={"help": "A list of pairs of integers which indicates a mapping\
                           from generation indices to token indices that will be forced before sampling."}
    )
    suppress_tokens: Optional[List[int]] = field(
        default=None,
        metadata={"help": "A list of tokens that will be suppressed at generation."}
    )
    max_new_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."}
    )