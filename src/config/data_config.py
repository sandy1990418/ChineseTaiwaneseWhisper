from dataclasses import dataclass, field
from typing import Optional, List, Literal
import os
import json


@dataclass
class DatasetAttr:
    load_from: str
    dataset_name: Optional[str] = None
    dataset_sha1: Optional[str] = None
    audio: Optional[str] = None
    target: Optional[str] = None
    language: Optional[str] = None
    dataset_args: Optional[list] = field(default_factory=lambda: list())
    dataset_kwargs: Optional[dict] = field(default_factory=lambda: dict(split="train"))

    def __repr__(self) -> str:
        return self.dataset_name


@dataclass
class DataArguments:
    dataset: Optional[str] = field(
        default=None,
        metadata={"help": "The name of provided dataset(s) to use. Use commas to separate multiple datasets."}
    )
    dataset_dir: Optional[str] = field(
        default="./youtube_data",
        metadata={"help": "The name of the folder containing datasets."}
    )
    sampling_rate: Optional[int] = field(
        default=16000,
        metadata={"help": "The frequency of the audio data."}
    )
    max_input_length: Optional[int] = field(
        default=30,
        metadata={"help": "Maximum input length in seconds for audio clips"}
    )
    do_lower_case: Optional[bool] = field(
        default=True,
        metadata={"help": "Lower case the target."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=1,
        metadata={"help": "The number of processes to use for the preprocessing."}
    )
    overwrite_cache: Optional[bool] = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number \
            of training examples to this value if set."}
    )
    max_eval_samples: Optional[float] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate the number of \
            evaluation examples to this value if set."}
    )
    test_dataset_name: Optional[str] = field(
        default="common_voice_13_test",
        metadata={"help": "Name of the dataset to use for testing. If not specified, a \
            portion of the train dataset will be used based on max_eval_samples."}
    )
    timestamp: Optional[bool] = field(
        default=False,
        metadata={"help": "Timestamp to control CommonVoice_13 add timestamp value."}
    )
    mix_strategy: Optional[Literal["concat", "interleave_under", "interleave_over"]] = field(
        default="concat",
        metadata={"help": "Strategy to use in dataset mixing."}
    )
    interleave_probs: Optional[str] = field(
        default=None,
        metadata={"help": "Probabilities to sample data from datasets. Use commas to separate multiple datasets."}
    )
    streaming: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable streaming mode."}
    )

    def init_for_training(self):
        dataset_names = [ds.strip() for ds in self.dataset.split(",")]
        dataset_info_path = os.path.join(self.dataset_dir, "dataset_info.json")
        
        if not os.path.exists(dataset_info_path):
            raise FileNotFoundError(f"dataset_info.json not found in {self.dataset_dir}. "
                                    f"Please ensure the file exists and the path is correct.")

        try:
            with open(dataset_info_path, "r") as f:
                dataset_info = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing dataset_info.json: {str(e)}. "
                             f"Please check if the JSON file is correctly formatted.")

        self.dataset_list: List[DatasetAttr] = []
        for name in dataset_names:
            if name not in dataset_info:
                raise ValueError(f"Dataset '{name}' not found in dataset_info.json. "
                                 f"Available datasets: {', '.join(dataset_info.keys())}")

            dataset_config = dataset_info[name]
            if "hf_hub_url" in dataset_config:
                dataset_attr = DatasetAttr("hf_hub", dataset_name=dataset_config["hf_hub_url"])
            elif "script_url" in dataset_config:
                dataset_attr = DatasetAttr("script", dataset_name=dataset_config["script_url"])
            elif "file_name" in dataset_config:
                dataset_attr = DatasetAttr("file", dataset_name=dataset_config["file_name"])
            else:
                raise ValueError(f"Invalid configuration for dataset '{name}'. "
                                 f"Must specify either 'hf_hub_url', 'script_url', or 'file_name'.")

            if "columns" in dataset_config:
                dataset_attr.audio = dataset_config["columns"].get("audio")
                dataset_attr.target = dataset_config["columns"].get("target")
                dataset_attr.language = dataset_config["columns"].get("language")
                dataset_attr.kwargs = dataset_config["columns"].get("kwargs", None)

            dataset_attr.dataset_args = dataset_config.get("dataset_args", [])
            dataset_attr.dataset_kwargs = dataset_config.get("dataset_kwargs", {})

            self.dataset_list.append(dataset_attr)
        if not self.dataset_list:
            raise ValueError("No valid datasets found in the configuration.")