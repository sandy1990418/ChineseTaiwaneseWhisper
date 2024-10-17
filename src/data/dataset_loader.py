from abc import ABC, abstractmethod
from datasets import (
    load_dataset,
    Dataset as HFDataset,
)
from src.config import DatasetAttr
import os
from src.utils.logging import logger
from src.data.dataset_preparation import DatasetPreparation
from src.config import DataArguments


MAX_DURATION_IN_SECONDS = 30.0
max_input_length = MAX_DURATION_IN_SECONDS * 16000


def is_audio_in_length_range(length):
    return 0 < length < max_input_length


def filter_labels(labels_length):
    return labels_length < 448


class DatasetStrategy(ABC):
    def __init__(self, dataset_attr: DatasetAttr, args: DataArguments):
        self.dataset_attr = dataset_attr
        self.args = args

    @abstractmethod
    def load_dataset(self, dataset_attr: DatasetAttr) -> HFDataset:
        raise NotImplementedError("You should implement this method")

    def preprocess_dataset(self) -> HFDataset:
        logger.info(f"Loading dataset from {self.dataset_attr.dataset_name}")
        dataset = self.load_dataset()
        logger.info(f"Preprocessing dataset from {self.dataset_attr.dataset_name}")
        self.dataset_preprocessor = DatasetPreparation()
        return self.dataset_preprocessor.preprocess(dataset, self.dataset_attr, self.args)


class HFHubDatasetStrategy(DatasetStrategy):
    def __init__(self, dataset_attr: DatasetAttr, args: DataArguments):
        super().__init__(dataset_attr, args)

    def load_dataset(self) -> HFDataset:
        logger.info("Loading dataset from Hugging Face Hub or Hugging Face Local")
        dataset = load_dataset(
            self.dataset_attr.dataset_name,
            *self.dataset_attr.dataset_args,
            **self.dataset_attr.dataset_kwargs,
        )
        if self.dataset_attr.kwargs:
            dataset = dataset.filter(lambda example: example["language"] == self.dataset_attr.kwargs.get('language'))
        return dataset
        

class LocalDatasetStrategy(DatasetStrategy):
    def __init__(self, dataset_attr: DatasetAttr, args: DataArguments):
        super().__init__(dataset_attr, args)
        
    def load_dataset(self) -> HFDataset:
        logger.info("Loading dataset from local directory")
        file_path = os.path.join(self.args.dataset_dir, self.dataset_attr.dataset_name)
        dataset = load_dataset('json', data_files=file_path, **self.dataset_attr.dataset_kwargs)
        return dataset


class DatasetLoaderFactory:
    def __init__(self, dataset_attr: DatasetAttr, args: DataArguments):
        self.dataset_attr = dataset_attr
        self.args = args

    def load_dataset(self) -> HFDataset:
        if self.dataset_attr.load_from == "hf_hub":
            return HFHubDatasetStrategy(self.dataset_attr, self.args).preprocess_dataset()
        else:
            return LocalDatasetStrategy(self.dataset_attr, self.args).preprocess_dataset()