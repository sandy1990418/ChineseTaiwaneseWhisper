from typing import Optional, Any, Dict, List, Union
import torch
import numpy as np
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import WhisperProcessor
import librosa
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChineseTaiwaneseDataset(Dataset):
    def __init__(self, 
                 dataset_name: str,
                 split: str,
                 processor: WhisperProcessor,
                 text_column: str = "sentence",
                 audio_column: str = "audio",
                 max_samples: Optional[int] = None,
                 language: str = "zh-TW"):
        self.dataset = load_dataset(dataset_name, "zh-TW", split=split)
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
        
        try:
            audio = item[self.audio_column]["array"]
            sampling_rate = item[self.audio_column]["sampling_rate"]
            
            # Resample audio to 16kHz if necessary
            if sampling_rate != 16000:
                audio = librosa.resample(y=audio, orig_sr=sampling_rate, target_sr=16000)
            
            # Process audio
            input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features

            # Process text
            text = item[self.text_column]
            if not isinstance(text, str):
                logger.warning(f"Text at index {idx} is not a string. Converting to string.")
                text = str(text)

            # Tokenize text
            labels = self.processor.tokenizer(text, return_tensors="pt").input_ids

            return {
                "input_features": input_features.squeeze(),
                "labels": labels.squeeze()
            }
        except Exception as e:
            logger.error(f"Error processing item at index {idx}: {e}")
            logger.error(f"Item contents: {item}")
            raise