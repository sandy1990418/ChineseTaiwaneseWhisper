from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any
import json
import soundfile as sf
from tqdm import tqdm
from datasets import Dataset, IterableDataset
from collections import defaultdict
from src.utils.logging import logger


class AudioSaver(ABC):
    @abstractmethod
    def save(self, audio: Any, sr: int, file_path: Path) -> None:
        raise NotImplementedError(
            "AudioSaver method must be implemented by a subclass."
        )


class JsonSaver(ABC):
    @abstractmethod
    def save(self, data: Any, file_path: Path) -> None:
        raise NotImplementedError("JsonSaver method must be implemented by a subclass.")


class HuggingFaceDatasetSaver(ABC):
    @abstractmethod
    def iterable_to_dataset(self, iterable_dataset: IterableDataset) -> Dataset:
        raise NotImplementedError(
            "HuggingFaceDatasetSaver method must be implemented by a subclass."
        )

    @abstractmethod
    def create_dataset_from_json(self, json_file: Path) -> Dataset:
        raise NotImplementedError(
            "HuggingFaceDatasetSaver method must be implemented by a subclass."
        )

    @abstractmethod
    def convert_dataset_to_json(self, dataset: Dataset, output_file: Path) -> None:
        raise NotImplementedError(
            "HuggingFaceDatasetSaver method must be implemented by a subclass."
        )


class SoundfileSaver(AudioSaver):
    def save(self, audio: Any, sr: int, file_path: str) -> None:
        sf.write(file_path, audio, sr)


class JsonAppendSaver(JsonSaver):
    def save(self, data: List[Dict], file_path: str) -> None:
        if Path(file_path).exists():
            with open(file_path, "r+", encoding="utf-8") as f:
                existing_data = json.load(f)
                existing_data.extend(data)
                f.seek(0)
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
                f.truncate()
        else:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)


class HuggingFaceDatasetSaver(HuggingFaceDatasetSaver):
    def iterable_to_dataset(self, iterable_dataset: IterableDataset) -> Dataset:
        data_dict = defaultdict(list)
        for item in iterable_dataset:
            for key, value in item.items():
                data_dict[key].append(value)
        return Dataset.from_dict(data_dict)

    def create_dataset_from_json(self, json_file: Path) -> Dataset:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Dataset.from_dict(
            {
                "client_id": [f"{Path(item['audio_path']).name}" for item in data],
                "path": [item["audio_path"] for item in data],
                "sentence": [item["text"] for item in data],
                "start": [item["start"] for item in data],
                "end": [item["end"] for item in data],
                "duration": [item["end"] - item["start"] for item in data],
            }
        )

    def convert_dataset_to_json(self, dataset: Dataset, output_file: Path) -> None:
        data = []
        for item in tqdm(dataset, desc="Converting dataset to JSON"):
            data.append(
                {
                    "client_id": item["client_id"],
                    "path": item["path"],
                    "sentence": item["sentence"],
                    "start": item["start"],
                    "end": item["end"],
                    "duration": item["duration"],
                }
            )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Dataset saved as JSON: {output_file}")