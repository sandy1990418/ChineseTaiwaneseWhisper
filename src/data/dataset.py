from typing import Optional, List, Union, Any
from torch.utils.data import Dataset, ConcatDataset
from datasets import load_dataset, load_from_disk
from transformers import WhisperProcessor
import librosa
import logging
import os
import json
# import multiprocessing
# from src.config.train_config import DataArguments


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChineseTaiwaneseDataset(Dataset):
    def __init__(self, 
                 dataset_names: Union[str, List[str]],
                 split: str,
                 processor: WhisperProcessor,
                 text_column: str = "sentence",
                 audio_column: str = "audio",
                 max_samples: Optional[int] = None,
                 dataset_config_names: Optional[Union[str, List[Any]]] = None,
                 use_timestamps: bool = False,
                 *args,
                 **kwargs):
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        if dataset_config_names is None:
            dataset_config_names = [None] * len(dataset_names)
        elif isinstance(dataset_config_names, str):
            dataset_config_names = [dataset_config_names]
        # Ensure dataset_config_names is at least as long as dataset_names
        if len(dataset_config_names) < len(dataset_names):
            dataset_config_names.extend([None] * (len(dataset_names) - len(dataset_config_names)))
        
        datasets = []

        for dataset_name, config_name in zip(dataset_names, dataset_config_names):
            try:
                dataset = load_dataset(dataset_name, config_name, split=split)
            except ValueError:
                dataset = load_from_disk(dataset_name)
            datasets.append(dataset)
        
        self.dataset = ConcatDataset(datasets)
        if max_samples is not None:
            self.dataset = [self.dataset[i] for i in range(min(max_samples, len(self.dataset)))]
        
        self.processor = processor
        self.text_column = text_column
        self.audio_column = audio_column
        self.use_timestamps = use_timestamps

        if use_timestamps:
            self.processor.tokenizer.predict_timestamps = True

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        
        try:
            # Handle both Common Voice and YouTube data formats
            if isinstance(item[self.audio_column], dict) and 'array' in item[self.audio_column]:
                # Common Voice format
                audio = item[self.audio_column]['array']
                sampling_rate = item[self.audio_column]['sampling_rate']
            else:
                # YouTube data format
                audio_path = item[self.audio_column]
                audio, sampling_rate = librosa.load(audio_path, sr=None)
            
            # Resample audio to 16kHz if necessary
            if sampling_rate != 16000:
                audio = librosa.resample(y=audio, orig_sr=sampling_rate, target_sr=16000)

            # Process audio
            input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features

            # Process text
            if self.use_timestamps:
                text = self._process_timestamps(item)
            else:
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
    
    def _process_timestamps(self, item):
        try:
            # Try to get the audio file path
            audio_path = item[self.audio_column]["path"]
            json_file = audio_path.replace(".wav", ".json")
            
            if os.path.exists(json_file):
                # YouTube-style timestamps (JSON file)
                transcript_data = self._load_json_robust(json_file)
                if not isinstance(transcript_data, type(None)):
                    formatted_transcript = ""
                    for segment in transcript_data:
                        start_time = segment.get("start", 0)
                        end_time = segment.get("end")
                        duration = segment.get("duration")
                        
                        if end_time is None and duration is not None:
                            end_time = start_time + duration
                        elif end_time is None and duration is None:
                            end_time = start_time + 5  # Assume 5 seconds if no duration info
                        
                        text = segment.get("text", "")
                        formatted_transcript += f"<|{start_time:.2f}|>{text}<|{end_time:.2f}|>"
                    
                    return formatted_transcript.strip()
                else:
                    # Fallback to using full audio duration if JSON parsing fails
                    return self._process_full_audio_timestamps(item)
            else:
                # Fallback to using full audio duration
                return self._process_full_audio_timestamps(item)
        
        except Exception as e:
            logger.error(f"Error processing timestamps: {str(e)}")
            # Fallback to using full audio duration
            return None

    def _load_json_robust(self, file_path):
        try:
            with open(file_path, 'rb', encoding='utf-8') as file:
                content = file.read()
                logger.debug(f"File content: {content[:100]}...")  # Log first 100 characters
                return json.loads(content)
        except FileNotFoundError:
            logger.warning(f"JSON file not found: {file_path}")
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing error in {file_path}: {str(e)}")
            logger.debug(f"Problematic content: {content[max(0, e.pos-20):e.pos+20]}")
        except UnicodeDecodeError as e:
            logger.warning(f"Unicode decode error in {file_path}: {str(e)}")
        except Exception as e:
            logger.warning(f"Unexpected error reading JSON file {file_path}: {str(e)}")
        return None

    def _process_full_audio_timestamps(self, item):
        try:
            audio = item[self.audio_column]["array"]
            sampling_rate = item[self.audio_column]["sampling_rate"]
            start_time = 0.0
            duration = len(audio) / sampling_rate
            text = item[self.text_column]
            return f"<|{start_time:.2f}|>{text}<|{duration:.2f}|>"
        except Exception as e:
            logger.error(f"Error processing full audio timestamps: {str(e)}")
            return item[self.text_column]