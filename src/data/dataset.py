import logging
from datasets import load_dataset, Dataset as HFDataset, concatenate_datasets, Audio, Features
from transformers import WhisperProcessor
from src.config import DataArguments, DatasetAttr
import librosa
import os
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChineseTaiwaneseDataset:
    def __init__(
        self,
        args: DataArguments,
        processor: WhisperProcessor,
        split: str = "train",
        **kwargs
    ):
        self.args = args
        self.processor = processor
        self.split = split
        if self.split == "train":
            self.args.init_for_training()
        else: 
            self.args.dataset = self.args.test_dataset_name
            self.args.init_for_training()
        self.dataset = self._load_and_prepare_dataset()
        self.kwargs = kwargs

    def _load_and_prepare_dataset(self) -> HFDataset:
        datasets = []

        for dataset_attr in self.args.dataset_list:
            logger.info(f"Loading dataset: {dataset_attr.dataset_name}")
            try:
                if dataset_attr.load_from == "hf_hub":
                    dataset = self._load_from_hf_hub(dataset_attr)
                elif dataset_attr.load_from in ["script", "file"]:
                    dataset = self._load_from_local(dataset_attr)
                else:
                    raise ValueError(f"Unknown load_from type: {dataset_attr.load_from}")

                logger.info(f"Dataset {dataset_attr.dataset_name} features: {dataset.features}")
                datasets.append(dataset)
            except Exception as e:
                logger.error(f"Error loading dataset {dataset_attr.dataset_name}: {str(e)}")
                continue

        if not datasets:
            raise ValueError("No datasets were successfully loaded.")

        logger.info("Concatenating datasets...")
        combined_dataset = concatenate_datasets(datasets)
        logger.info(f"Combined dataset features: {combined_dataset.features}")
        combined_dataset = combined_dataset.shuffle(seed=42)
        if self.split == 'test':
            combined_dataset = combined_dataset.select(range(1000))

        return combined_dataset.map(
            self._preprocess_function,
            remove_columns=combined_dataset.column_names,
            num_proc=self.args.preprocessing_num_workers,
            desc="Running preprocessor on dataset",
            batched=False,
        )

    def _load_from_hf_hub(self, dataset_attr: DatasetAttr) -> HFDataset:
        dataset = load_dataset(
            dataset_attr.dataset_name,
            *dataset_attr.dataset_args,
            **dataset_attr.dataset_kwargs,
        )
        return self._prepare_dataset(dataset, dataset_attr)

    def _load_from_local(self, dataset_attr: DatasetAttr) -> HFDataset:
        file_path = os.path.join(self.args.dataset_dir, dataset_attr.dataset_name)
        dataset = load_dataset('json', data_files=file_path, **dataset_attr.dataset_kwargs)
        return self._prepare_dataset(dataset, dataset_attr)

    def _prepare_dataset(self, dataset: HFDataset, dataset_attr: DatasetAttr) -> HFDataset:
        column_mapping = {
            dataset_attr.audio: "audio",
            dataset_attr.target: "target",
        }
        dataset = dataset.rename_columns(column_mapping)
        # Ensure all audio is in the correct format and sampling rate
        target_sampling_rate = self.args.sampling_rate
        
        if isinstance(dataset.features["audio"], Audio):
            current_sampling_rate = dataset.features["audio"].sampling_rate
            if current_sampling_rate != target_sampling_rate:
                logger.info(f"Resampling audio from {current_sampling_rate} Hz to {target_sampling_rate} Hz")
                dataset = dataset.cast_column("audio", Audio(sampling_rate=target_sampling_rate))
        else:
            logger.info(f"Converting audio column to Audio feature with {target_sampling_rate} Hz sampling rate")
            dataset = dataset.cast_column("audio", Audio(sampling_rate=target_sampling_rate))
        
        # Ensure all datasets have the same feature structure
        target_features = Features({
            'audio': Audio(sampling_rate=target_sampling_rate),
            'target': dataset.features["target"],
        })
        columns_to_remove = [col for col in dataset.column_names if col not in ["audio", "target"]]
        dataset = dataset.map(remove_columns=columns_to_remove)
        dataset = dataset.cast(target_features)
        self.language = dataset_attr.language
        return dataset

    def _preprocess_function(self, example):
        try:
            audio = example["audio"]
            if isinstance(audio, dict):
                audio_array = audio["array"]
                sampling_rate = audio["sampling_rate"]
            # elif isinstance(audio, str):
            #     # If audio is a file path, load it
            #     audio_array, sampling_rate = librosa.load(audio, sr=self.args.sampling_rate)
            else:
                raise ValueError(f"Unexpected audio type: {type(audio)}")

            # Resample if necessary
            if sampling_rate != self.args.sampling_rate:
                audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=self.args.sampling_rate)

            # Ensure audio doesn't exceed max_input_length
            max_samples = int(self.args.max_input_length * self.args.sampling_rate)
            audio_array = audio_array[:max_samples]
            
            self.processor.tokenizer.set_prefix_tokens(language=self.language)
            input_features = self.processor.feature_extractor(
                audio_array,
                sampling_rate=self.args.sampling_rate,
                return_tensors="pt",
            ).input_features[0]

            target_text = example["target"]
            if not self.args.timestamp and self.args.do_lower_case:
                target_text = target_text.lower()        
            
            if self.args.timestamp:
                if isinstance(target_text, list):
                    processed_text = ""
                    for segment in target_text:
                        start_time = segment['start']
                        end_time = segment['end']
                        text = segment['text']
                        if self.args.do_lower_case:
                            text = text.lower()    
                        processed_text += f"<|{start_time:.2f}|>{text}<|{end_time:.2f}|>"
                    target_text = processed_text
                else:
                    audio_length = len(audio_array) / self.args.sampling_rate
                    target_text = f"<|0.00|>{target_text}<|{audio_length:.2f}|>"

            # Tokenize the processed text
            self.processor.tokenizer.predict_timestamps = self.args.timestamp

            labels = self.processor.tokenizer(
                target_text,
                return_tensors="pt",
            ).input_ids.squeeze()  # Remove batch dimension

            # Check for NaN or infinity values
            if torch.isnan(input_features).any() or torch.isinf(input_features).any():
                raise ValueError("NaN or infinity values detected in input_features")

            if torch.isnan(labels).any() or torch.isinf(labels).any():
                raise ValueError("NaN or infinity values detected in labels")

            return {
                "input_features": input_features,
                "labels": labels
            }
        except Exception as e:
            logger.error(f"Error preprocessing example: {str(e)}")
            return None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    @classmethod
    def create_train_and_test_datasets(
            cls, 
            args: DataArguments, 
            processor: WhisperProcessor, 
            **kwargs):
        if args.test_dataset_name:
            train_dataset = cls(args, processor, split="train", **kwargs)
            test_dataset = cls(args, processor, split="test", **kwargs)

        else: 
            if not 0 < args.max_eval_samples < 1:
                raise ValueError("When no test_dataset_name is specified, \
                    max_eval_samples must be a float between 0 and 1, representing \
                    the ratio of the dataset to use for evaluation.")

            full_dataset = cls(args, processor, **kwargs)
            
            total_samples = len(full_dataset.dataset)
            max_eval_samples = int(total_samples * args.max_eval_samples)
            max_train_samples = total_samples - max_eval_samples
            train_dataset = full_dataset.select(range(max_train_samples))
            test_dataset = full_dataset.select(range(max_train_samples, total_samples))
        
        return train_dataset, test_dataset