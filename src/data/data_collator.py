from dataclasses import dataclass
from typing import Dict, List
import torch
from transformers import WhisperProcessor

@dataclass
class WhisperDataCollator:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_features = [feature["input_features"] for feature in features]
        labels = [feature["labels"] for feature in features]

        input_features = torch.stack(input_features)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

        return {
            "input_features": input_features,
            "labels": labels,
        }