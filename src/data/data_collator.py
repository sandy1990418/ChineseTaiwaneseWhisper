from dataclasses import dataclass
from typing import Dict, List
import torch
from src.utils import logger
from dataclasses import dataclass
from typing import Any, Union


@dataclass
class WhisperDataCollator:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        model_input_name = self.processor.model_input_names[0]
        input_features = [
            {model_input_name: feature[model_input_name]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


# @dataclass
# class WhisperDataCollator:
#     processor: WhisperProcessor

#     def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
#         try:
#             input_features = [feature["input_features"] for feature in features]
#             labels = [feature["labels"] for feature in features]

#             input_features = torch.stack(input_features)
#             labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

#             batch = {
#                 "input_features": input_features,
#                 "labels": labels,
#             }

#             return batch
#         except Exception as e:
#             logger.error(f"Error in data collator: {e}")
#             logger.error(f"Features causing the error: {features}")
#             raise
