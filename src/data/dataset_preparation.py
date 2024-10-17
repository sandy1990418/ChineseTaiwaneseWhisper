from abc import ABC, abstractmethod
from typing import Any
from datasets import Dataset as HFDataset, Audio, Features, Value
from functools import partial
from src.config import DatasetAttr
from src.utils import logger
from src.config import DataArguments
import librosa


class DatasetPreparationStrategy(ABC):
    @abstractmethod
    def prepare(self) -> Any:
        raise NotImplementedError("You should implement this method")


class AudioDatasetPreparationStrategy(DatasetPreparationStrategy):
    def __init__(self, dataset: HFDataset, attr: DatasetAttr, args: DataArguments):
        self.args = args
        self.attr = attr
        self.dataset = dataset

    def prepare(self) -> HFDataset:
        self._rename_columns()
        self._resample_audio()
        self._adjust_target()
        self._standardize_features()
        return self.dataset

    def _rename_columns(self):
        column_mapping = {
            self.attr.audio: "audio",
            self.attr.target: "target",
        }
        self.dataset = self.dataset.rename_columns(column_mapping)

    def _resample_audio(self):
        target_sampling_rate = self.args.sampling_rate
        if isinstance(self.dataset.features["audio"], Audio):
            current_sampling_rate = self.dataset.features["audio"].sampling_rate
            if current_sampling_rate != target_sampling_rate:
                logger.info(
                    f"Resampling audio from {current_sampling_rate} Hz to {target_sampling_rate} Hz"
                )
                self.dataset = self.dataset.cast_column(
                    "audio", Audio(sampling_rate=target_sampling_rate)
                )
        else:
            logger.info(
                f"Converting audio column to Audio feature with {target_sampling_rate} Hz sampling rate"
            )
            self.dataset = self.dataset.cast_column(
                "audio", Audio(sampling_rate=target_sampling_rate)
            )

    def _adjust_target_to_list(self, data, language):
        if language:
            data["language"] = language
        else:
            raise ValueError("You should provide language for dataset")
        if isinstance(data["target"], str):
            
            if not self.args.timestamp:
                data["target"] = [
                    {"start": 0.0, "end": 0.0, "text": data["target"]}
                ]
            else:
                try:
                    # TODO: 這邊有問題，不知道為什麼抓不到local的檔案
                    data["target"] = [
                        {
                            "start": 0.0,
                            "end": float(
                                round(
                                    librosa.get_duration(
                                        path=data["audio"]["path"]
                                    ),
                                    2,
                                )
                            ),
                            "text": data["target"],
                        }
                    ]
                except FileNotFoundError:
                    data["target"] = [
                        {
                            "start": 0.0,
                            "end": round(data["duration"], 2),
                            "text": data["target"],
                        }
                    ]
        elif isinstance(data["target"], list):
            data["target"] = [
                {
                    "start": float(target["start"]),
                    "end": float(target["end"]),
                    "text": target["text"],
                }
                for target in data["target"]
            ]
        else:
            raise ValueError(
                f"Only support `target` type of [list, str] but get {type(data['target'])}"
            )
        return data

    def _adjust_target(self):
        self.dataset = self.dataset.map(
            partial(self._adjust_target_to_list, **{"language": self.attr.language}),
            num_proc=self.args.preprocessing_num_workers,
            desc="Running preprocessor on dataset",
            load_from_cache_file=not self.args.overwrite_cache,
        )

    def _standardize_features(self):
        target_features = Features(
            {
                "audio": Audio(sampling_rate=self.args.sampling_rate),
                "target": [
                    {
                        "end": Value(dtype="float64", id=None),
                        "start": Value(dtype="float64", id=None),
                        "text": Value(dtype="string", id=None),
                    }
                ],
            }
        )

        columns_to_remove = [
            col for col in self.dataset.column_names if col not in ["audio", "target"]
        ]
        self.dataset = self.dataset.map(remove_columns=columns_to_remove)
        self.dataset = self.dataset.cast(target_features)


class DatasetPreparationStrategyFactory:
    @staticmethod
    def create_strategy(
        dataset: HFDataset, dataset_attr: DatasetAttr, args: DataArguments
    ) -> DatasetPreparationStrategy:
        if dataset_attr.audio == "audio" or dataset_attr.audio == "audio_path":
            return AudioDatasetPreparationStrategy(dataset, dataset_attr, args)
        # Add more conditions for other types of datasets
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_attr}")


class DatasetPreparation:
    def preprocess(
        self, dataset: HFDataset, dataset_attr: DatasetAttr, args: DataArguments
    ) -> HFDataset:
        strategy = DatasetPreparationStrategyFactory.create_strategy(
            dataset, dataset_attr, args
        )
        return strategy.prepare()
