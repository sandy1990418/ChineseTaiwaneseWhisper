import pytest
import torch
from src.models.whisper_model import load_whisper_model


@pytest.fixture
def model_and_processor():
    return load_whisper_model("openai/whisper-small")


def test_model_output(model_and_processor):
    model, processor = model_and_processor
    input_features = torch.rand(1, 80, 3000)
    input_features = input_features.to(model.device)
    
    with torch.no_grad():
        output = model(input_features)
    
    assert output is not None
    assert hasattr(output, 'logits')
    assert output.logits.dim() == 3