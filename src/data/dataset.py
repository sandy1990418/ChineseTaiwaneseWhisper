from typing import Optional
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import WhisperProcessor

class ChineseTaiwaneseDataset(Dataset):
    def __init__(self, 
                 dataset_name: str = "mozilla-foundation/common_voice_11_0", 
                 split: str = "train",
                 processor: WhisperProcessor = None,
                 text_column: str = "sentence",
                 audio_column: str = "audio",
                 max_samples: Optional[int] = None,
                 language: str = "zh-TW"):
        self.dataset = load_dataset(dataset_name, language, split=split)
        if max_samples is not None:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        self.processor = processor
        self.text_column = text_column
        self.audio_column = audio_column
        self.language = language

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        audio = item[self.audio_column]["array"]
        
        input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        
        with self.processor.as_target_processor():
            labels = self.processor(text=item[self.text_column], return_tensors="pt").input_ids

        return {
            "input_features": input_features.squeeze(),
            "labels": labels.squeeze()
        }