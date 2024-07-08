import pytest
from src.data.dataset import ChineseTaiwaneseDataset
from transformers import WhisperProcessor


@pytest.fixture
def processor():
    return WhisperProcessor.from_pretrained("openai/whisper-small")


@pytest.fixture
def dataset(processor):
    return ChineseTaiwaneseDataset("mozilla-foundation/common_voice_11_0", "test", processor, max_samples=10)


def test_dataset_length(dataset):
    assert len(dataset) == 10


def test_dataset_item(dataset):
    item = dataset[0]
    assert "input_features" in item
    assert "labels" in item
    assert item["input_features"].dim() == 2
    assert item["labels"].dim() == 1