from datasets import (
    Dataset as HFDataset,
    concatenate_datasets,
    interleave_datasets,
)
from transformers import WhisperProcessor
from src.config import DataArguments
from typing import List
import re
from src.data import DatasetLoaderFactory, PreprocessingPipeline
from src.utils.logging import logger

MAX_DURATION_IN_SECONDS = 30.0
max_input_length = MAX_DURATION_IN_SECONDS * 16000


def is_audio_in_length_range(length):
    return 0 < length < max_input_length


def filter_labels(labels_length):
    """Filter label sequences longer than max length (448)"""
    return labels_length < 448


class ChineseTaiwaneseDataset:
    def __init__(
        self,
        args: DataArguments,
        processor: WhisperProcessor,
        split: str = "train",
        dataset_loader: DatasetLoaderFactory = DatasetLoaderFactory,
        preprocessing_pipeline: PreprocessingPipeline = PreprocessingPipeline,
        **kwargs,
    ):
        self.args = args
        self.processor = processor
        self.split = split
        if self.split == "train":
            self.args.init_for_training()
        else:
            self.args.dataset = self.args.test_dataset_name
            self.args.init_for_training()
        self.dataset_loader = dataset_loader
        self.preprocessing_pipeline = preprocessing_pipeline
        self.dataset = self._load_and_prepare_dataset()
        self.kwargs = kwargs

    def _load_and_prepare_dataset(self) -> HFDataset:
        datasets = []

        for dataset_attr in self.args.dataset_list:
            logger.info(f"Current Dataset: {dataset_attr}")
            strategy = self.dataset_loader(dataset_attr, self.args)
            dataset = strategy.load_dataset()
            datasets.append(dataset)

        combined_dataset = self._concate_dataset(datasets)
        column_names = list(next(iter(combined_dataset)).keys())
        self.language = dataset_attr.language

        pipeline = self.preprocessing_pipeline(self.args, self.processor, self.language)
        dataset = dataset.map(
            pipeline.process,
            remove_columns=column_names,
            num_proc=self.args.preprocessing_num_workers,
            desc="Running preprocessor on dataset",
            batched=False,
            load_from_cache_file=not self.args.overwrite_cache,
        )

        combined_dataset = self._apply_filters(combined_dataset)
        return combined_dataset

    def _apply_filters(self, dataset: HFDataset):
        dataset = dataset.filter(
            is_audio_in_length_range, input_columns=["input_length"]
        )
        dataset = dataset.filter(filter_labels, input_columns=["label_length"])

        return dataset

    def _concate_dataset(self, datasets):
        if not datasets:
            raise ValueError("No datasets were successfully loaded.")

        logger.info("Concatenating datasets...")
        combined_dataset = concatenate_datasets(datasets)

        if len(self.args.dataset) == 1:
            combined_dataset = datasets
        elif self.args.mix_strategy == "concat":
            if self.args.streaming:
                logger.warning(
                    "The samples between different datasets will not be mixed in streaming mode."
                )
            combined_dataset = concatenate_datasets(datasets)
        elif self.args.mix_strategy.startswith("interleave"):
            if not self.args.streaming:
                logger.warning(
                    "We recommend using `mix_strategy=concat` in non-streaming mode."
                )
            stopping_strategy = (
                "first_exhausted"
                if self.args.mix_strategy.endswith("under")
                else "all_exhausted"
            )
            combined_dataset = interleave_datasets(
                datasets,
                self.args.interleave_probs,
                stopping_strategy=stopping_strategy,
            )
        else:
            raise ValueError("Unknown mixing strategy.")

        logger.info(f"Combined dataset features: {combined_dataset.features}")
        combined_dataset = combined_dataset.shuffle(seed=42)

        if self.split == "test":
            combined_dataset = combined_dataset.select(range(100))

        return combined_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    @classmethod
    def create_train_and_test_datasets(
        cls, args: DataArguments, processor: WhisperProcessor, **kwargs
    ):
        if args.test_dataset_name:
            train_dataset = cls(args, processor, split="train", **kwargs)
            test_dataset = cls(args, processor, split="test", **kwargs)

        else:
            if not 0 < args.max_eval_samples < 1:
                raise ValueError(
                    "When no test_dataset_name is specified, \
                    max_eval_samples must be a float between 0 and 1, representing \
                    the ratio of the dataset to use for evaluation."
                )

            full_dataset = cls(args, processor, **kwargs)

            total_samples = len(full_dataset.dataset)
            max_eval_samples = int(total_samples * args.max_eval_samples)
            max_train_samples = total_samples - max_eval_samples
            train_dataset = full_dataset.select(range(max_train_samples))
            test_dataset = full_dataset.select(range(max_train_samples, total_samples))

        return train_dataset, test_dataset


def remove_punctuation(text: str or List[str]):
    punctuation = "!,.;:?、！，。；：？"
    if isinstance(text, str):
        text = re.sub(r"[{}]+".format(punctuation), "", text).strip()
        return text
    elif isinstance(text, list):
        result_text = []
        for t in text:
            t = re.sub(r"[{}]+".format(punctuation), "", t).strip()
            result_text.append(t)
        return result_text
    else:
        raise Exception(f"Not support this type {type(text)}")
