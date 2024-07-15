from torch.utils.data import Dataset
from datasets import load_dataset, load_from_disk, concatenate_datasets
from transformers import WhisperProcessor
import librosa
import logging
from typing import Union, List, Optional, Any, Callable

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChineseTaiwaneseDataset(Dataset):
    def __init__(
        self,
        dataset_names: Union[str, List[str]],
        split: str,
        processor: WhisperProcessor,
        text_column: str = "sentence",
        audio_column: str = "audio",
        max_samples: Optional[int] = None,
        dataset_config_names: Optional[Union[str, List[Any]]] = None,
        use_timestamps: bool = False,
        processor_config: Callable = None,
        test_size: float = 0.002,
        random_seed: int = 42,
        *args,
        **kwargs,
    ):

        self.processor = processor
        self.text_column = text_column
        self.audio_column = audio_column
        self.use_timestamps = use_timestamps
        self.processor_config = processor_config

        self.dataset = self._load_and_prepare_dataset(
            dataset_names,
            dataset_config_names,
            split,
            max_samples,
            test_size,
            random_seed,
        )

        if use_timestamps:
            self.processor.tokenizer.predict_timestamps = True

    def _load_and_prepare_dataset(
        self,
        dataset_names,
        dataset_config_names,
        split,
        max_samples,
        test_size,
        random_seed,
    ):
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        if dataset_config_names is None:
            dataset_config_names = [None] * len(dataset_names)
        elif isinstance(dataset_config_names, str):
            dataset_config_names = [dataset_config_names]

        if len(dataset_config_names) < len(dataset_names):
            dataset_config_names.extend(
                [None] * (len(dataset_names) - len(dataset_config_names))
            )

        datasets = []
        for dataset_name, config_name in zip(dataset_names, dataset_config_names):
            logger.info(f"Loading dataset: {dataset_name}")
            try:
                dataset = load_dataset(dataset_name, config_name, split=split)
            except ValueError:
                dataset = load_from_disk(dataset_name)
            datasets.append(dataset)

        combined_dataset = concatenate_datasets(datasets)
        combined_dataset = combined_dataset.shuffle(seed=random_seed)

        if max_samples is not None:
            combined_dataset = combined_dataset.select(
                range(min(max_samples, len(combined_dataset)))
            )

        # Preprocess the entire dataset
        combined_dataset = combined_dataset.map(
            self._preprocess_function,
            remove_columns=combined_dataset.column_names,
            num_proc=4,  # Adjust based on your CPU cores
            batched=True,
            batch_size=4,  # Adjust based on your memory constraints
        )

        # test_size = int(len(combined_dataset) * test_size)
        # train_size = len(combined_dataset) - test_size
        # train_dataset = combined_dataset.select(range(train_size))
        # test_dataset = combined_dataset.select(range(train_size, len(combined_dataset)))

        return combined_dataset

    def _preprocess_function(self, examples):
        audio_arrays = []
        for idx, audio in enumerate(examples[self.audio_column]):
            if isinstance(audio, dict) and "array" in audio:
                audio_arrays.append(audio["array"])
            else:
                audio_array, _ = librosa.load(audio, sr=16000)
                audio_arrays.append(audio_array[examples[idx]["start"]*16000: examples[idx]["end"]*16000])

        inputs = self.processor(
            audio_arrays, sampling_rate=16000, return_tensors="pt", padding=True
        )
        breakpoint()
        with self.processor.as_target_processor():
            labels = self.processor(
                examples[self.text_column], return_tensors="pt", padding=True
            )

        inputs["labels"] = labels["input_ids"]

        return inputs

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {"input_features": item["input_features"], "labels": item["labels"]}

    def get_test_dataset(self):
        return self.test_dataset
