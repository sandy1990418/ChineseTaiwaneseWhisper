from abc import abstractmethod, ABC
from datasets import Dataset as HFDataset
from src.config import DataArguments
from src.utils import logger
from transformers import WhisperProcessor
import librosa
import torch
from typing import List, Dict, Any


class PreporcessorStrategy(ABC):
    @abstractmethod
    def process(self):
        raise NotImplementedError(
            "This method `PreporcessorStrategy` should be implemented."
        )


class AudioPreprocessor(PreporcessorStrategy):
    def process(
        self,
        dataset: HFDataset,
        args: DataArguments,
        processor: WhisperProcessor,
        language: str,
        **kwargs,
    ) -> HFDataset:
        try:
            audio = dataset["audio"]
            if isinstance(audio, dict):
                audio_array = audio["array"]
                sampling_rate = audio["sampling_rate"]
            else:
                raise ValueError(f"Unexpected audio type: {type(audio)}")

            if sampling_rate != args.sampling_rate:
                audio_array = librosa.resample(
                    audio_array, orig_sr=sampling_rate, target_sr=args.sampling_rate
                )

            processor.tokenizer.set_prefix_tokens(
                language=language,
                task="transcribe",
                predict_timestamps=args.timestamp,
            )

            dataset["input_features"] = processor.feature_extractor(
                audio_array,
                sampling_rate=args.sampling_rate,
                return_tensors="pt",
            ).input_features[0]

            dataset["input_length"] = audio_array.shape[0] / args.sampling_rate
            return dataset

        except Exception as e:
            logger.error(f"Error preprocessing example: {str(e)}")
            raise ValueError("`AudioPreprocessor` error raise")


class TextPreprocessor(PreporcessorStrategy):
    def process(
        self,
        dataset: HFDataset,
        args: DataArguments,
        processor: WhisperProcessor,
        **kwargs,
    ) -> HFDataset:
        try:
            target_text = dataset["target"]
            if not args.timestamp and isinstance(target_text, list):
                target_text = " ".join(target_text)

            if args.timestamp:
                target_text = self._process_timestamp(target_text, dataset, args)

            if args.do_lower_case:
                target_text = target_text.lower()

            # Tokenize the processed text
            # self.processor.tokenizer.predict_timestamps = self.args.timestamp
            dataset["labels"] = processor.tokenizer(
                target_text, return_tensors="pt", add_special_tokens=True
            ).input_ids[
                0
            ]  # decode_with_timestamps Remove batch dimension

            dataset["labels_length"] = len(
                processor.tokenizer(target_text, add_special_tokens=True).input_ids
            )
        except Exception as e:
            logger.error(f"Error preprocessing example: {str(e)}")
            raise ValueError("`TextPreprocessor` error raise")

        return dataset

    def _process_timestamp(self, target_text, dataset, args):
        if isinstance(target_text, list):
            processed_text = ""
            for segment in target_text:
                start_time = segment["start"]
                end_time = segment["end"]
                text = segment["text"]
                processed_text += f"<|{start_time:.2f}|>{text}<|{end_time:.2f}|>"
            target_text = processed_text
        else:
            audio = dataset["audio"]
            audio_array = audio["array"]
            audio_length = len(audio_array) / args.sampling_rate
            target_text = f"<|0.00|>{target_text}<|{audio_length:.2f}|>"

        return target_text


class ValidationStep(PreporcessorStrategy):
    def process(self, dataset: Dict[str, Any], **kwargs) -> HFDataset:
        if (
            torch.isnan(dataset["input_features"]).any()
            or torch.isinf(dataset["input_features"]).any()
        ):
            raise ValueError("NaN or infinity values detected in input_features")

        if torch.isnan(dataset["labels"]).any() or torch.isinf(dataset["labels"]).any():
            raise ValueError("NaN or infinity values detected in labels")

        return dataset


class PipelineStrategy(ABC):
    def __init__(self, args, processor, language):
        self.args = args
        self.processor = processor
        self.language = language
        self.steps: List[PreporcessorStrategy] = []

    @abstractmethod
    def create_pipeline(self) -> None:
        pass

    def add_step(self, step: PreporcessorStrategy) -> None:
        self.steps.append(step)

    @abstractmethod
    def process(self, dataset: Any) -> Any:
        pass


class PreprocessingPipeline(PipelineStrategy):
    def __init__(self, args: DataArguments, processor: WhisperProcessor, language: str):
        self.args = args
        self.processor = processor
        self.language = language

    def create_pipeline(self) -> List[PreporcessorStrategy]:
        self.steps = [AudioPreprocessor(), TextPreprocessor(), ValidationStep()]

    def process_single(self, dataset: HFDataset) -> HFDataset:
        for step in self.steps:
            param = {
                "dataset": dataset,
                "args": self.args,
                "processor": self.processor,
                "language": self.language,
            }
            dataset = step.process(**param)
        return dataset

    def process(self, dataset: HFDataset) -> Dict:
        self.create_pipeline()
        dataset = self.process_single(dataset)
        return {
            "input_features": dataset["input_features"],
            "labels": dataset["labels"],
            "input_length": dataset["input_length"],
            "labels_length": dataset["labels_length"],
        }
