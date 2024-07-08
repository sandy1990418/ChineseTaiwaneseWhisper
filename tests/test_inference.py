import pytest
import numpy as np
from src.inference.flexible_inference import ChineseTaiwaneseASRInference


@pytest.fixture
def inference():
    return ChineseTaiwaneseASRInference("openai/whisper-small", 
                                        device="cpu", 
                                        use_faster_whisper=False, 
                                        language="zh-TW")


def test_transcribe_stream(inference):
    def dummy_audio_stream():
        for _ in range(5):  # 5 chunks of 1 second each
            yield np.zeros(16000, dtype=np.float32)

    transcriptions = list(inference.transcribe_stream(dummy_audio_stream(), chunk_duration=1))
    
    assert len(transcriptions) > 0
    for transcription in transcriptions:
        assert isinstance(transcription, str)