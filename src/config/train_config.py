from dataclasses import dataclass, field, asdict
from typing import Optional, List, Union, Literal, Dict, Any
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
        default=16,
        metadata={"help": "LoRA attention dimension"}
    )
    lora_alpha: int = field(
        default=64,
        metadata={"help": "LoRA alpha"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout"}
    )


@dataclass
class WhisperTrainingArguments(Seq2SeqTrainingArguments):
    output_dir: str = field(
        default="./whisper-finetuned",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    overwrite_output_dir: bool = field(
        default=True,
        metadata={"help": "If True, overwrite the content of the output directory. Use this to continue training\
                   if output_dir points to a checkpoint directory."},
    )
    # auto_find_batch_size_size: bool = field(
    #     default=True, 
    #     metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    # )
    per_device_train_batch_size: int = field(
        default=16, 
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    gradient_accumulation_steps: int = field(
        default=4,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."}
    )
    num_train_epochs: float = field(
        default=3.0, 
        metadata={"help": "Total number of training epochs to perform."}
    )
    warmup_steps: int = field(
        default=100, 
        metadata={"help": "Linear warmup over warmup_steps."}
    )
    warmup_ratio: float = field(
        default=0.1, 
        metadata={"help": "Linear warmup over warmup_ratio."}
    )
    lr_scheduler_type: str = field(
        default="linear", 
        metadata={"help": "lr_scheduler_types."}
    )
    learning_rate: float = field(
        default=1e-5, 
        metadata={"help": "The initial learning rate for AdamW."}
    )
    fp16: bool = field(
        default=True,
        metadata={"help": "Whether to use 16-bit (mixed) precision instead of 32-bit"}
    )
    eval_strategy: str = field(
        default="steps",
        metadata={"help": "The evaluation strategy to use."}
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "The saving strategy to use."}
    )
    save_steps: int = field(
        default=50,
        metadata={"help": "Save checkpoint every X updates steps."}
    )
    eval_steps: int = field(
        default=50,
        metadata={"help": "Run an evaluation every X steps."}
    )
    logging_steps: int = field(
        default=50,
        metadata={"help": "Log every X updates steps."}
    )
    save_total_limit: Optional[int] = field(
        default=-1,
        metadata={"help": "Limit the total amount of checkpoints."}
    )
    # https://github.com/huggingface/blog/issues/933
    # metric_for_best_model: str = field(
    #     default="loss",
    #     metadata={"help": "The metric to use to compare two different models."}
    # )
    # greater_is_better: bool = field(
    #     default=False,
    #     metadata={"help": "Whether the `metric_for_best_model` should be maximized or not."}
    # )
    load_best_model_at_end: bool = field(
        default=False,
        metadata={"help": "Whether or not to load the best model found during training at the end of training."}
    )
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."}
    )
    dataloader_num_workers: int = field(
        default=1,
        metadata={"help": "Number of subprocesses to use for data loading."}
    )
    ddp_find_unused_parameters: Optional[bool] = field(
        default=False,
        metadata={"help": "When using distributed training, the value of the flag \
            `find_unused_parameters` passed to `DistributedDataParallel`."}
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."}
    )
    dataloader_pin_memory: bool = field(
        default=True,
        metadata={"help": "If True, use dataloader_pin_memory"}
    )    
    predict_with_generate: bool = field(
        default=True,
        metadata={"help": "If True, use predict_with_generate"}
    )    
    # max_grad_norm: float = field(
    #     default=1.0, 
    #     metadata={"help": "max_grad_norm."}
    # )
    eval_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing \
            a backward/update pass in evaluation."}
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={"help": "Number of subprocesses to use for data loading (PyTorch only). \
            0 means that the data will be loaded in the main process."}
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Whether or not to automatically remove the columns unused by the model forward method."}
    )
    label_names: str = field(
        default="labels",
        metadata={"help": "Whether or not to automatically remove the columns unused by the model forward method."}
    )
    # predict_with_generate: bool = field(
    #     default=False,
    #     metadata={"help": "Whether or not to automatically remove the columns unused by the model forward method."}
    # )
    return_loss: bool = field(
        default=True,
        metadata={"help": "Whether or not to return loss."}
    )
    

@dataclass
class WhisperProcessorConfig:
    # Feature Extractor Arguments
    feature_size: int = field(
        default=80,
        metadata={"help": "The feature dimension of the extracted features."}
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
        default=True,
        metadata={"help": "Whether to return timestamps in the decoded output."}
    )
    model_max_length: Optional[int] = field(
        default=1024,
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


@dataclass
class WhisperPredictionArguments:
    do_sample: Optional[bool] = field(
        default=True,
        metadata={"help": "Whether or not to use sampling, use greedy decoding otherwise."}
    )
    temperature: Optional[float] = field(
        default=0.95,
        metadata={"help": "The value used to modulate the next token probabilities."}
    )
    top_p: Optional[float] = field(
        default=0.7,
        metadata={"help": "The smallest set of most probable tokens with probabilities \
                  that add up to top_p or higher are kept."}
    )
    top_k: Optional[int] = field(
        default=50,
        metadata={"help": "The number of highest probability vocabulary tokens to keep for top-k filtering."}
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={"help": "Number of beams for beam search. 1 means no beam search."}
    )
    # max_length: Optional[int] = field(
    #     default=None,
    #     metadata={"help": "The maximum length the generated tokens can have. It can be overridden by max_new_tokens."}
    # )
    # max_new_tokens: Optional[int] = field(
    #     default=512,
    #     metadata={"help": "The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt."}
    # )
    repetition_penalty: Optional[float] = field(
        default=1.0,
        metadata={"help": "The parameter for repetition penalty. 1.0 means no penalty."}
    )
    length_penalty: Optional[float] = field(
        default=1.0,
        metadata={"help": "Exponential penalty to the length that is used with beam-based generation."}
    )
    metric: Optional[Literal["wer", "cer"]] = field(
        default="cer",
        metadata={"help": "metric for hugging face evaluate module."}
    )

    def to_dict(self) -> Dict[str, Any]:
        args = asdict(self)
        # if args.get("max_new_tokens", None):
        #     args.pop("max_length", None)
        return args
