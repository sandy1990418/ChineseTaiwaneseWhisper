from dataclasses import dataclass, field


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
