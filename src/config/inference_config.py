from dataclasses import dataclass, field
from typing import List
import torch
import os


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
