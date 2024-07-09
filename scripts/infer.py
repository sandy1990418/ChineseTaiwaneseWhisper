from transformers import HfArgumentParser
import numpy as np 
from typing import Dict, List
import soundfile as sf
from src.inference.flexible_inference import ChineseTaiwaneseASRInference
import logging
import os
import json
from src.config.train_config import InferenceArguments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_audio(file_path):
    audio, sample_rate = sf.read(file_path)
    # Convert to mono if stereo
    # Check if the audio is stereo
    if audio.ndim == 2 and audio.shape[1] == 2:
        return audio[:, 0], audio[:, 1], sample_rate
    else:
        return audio, sample_rate


def transcribe_channel(inference, audio: np.ndarray, sample_rate: int, channel_name: str) -> Dict:
    """
    Transcribe a single audio channel.
    
    Args:
    inference: The inference object with a transcribe_batch method.
    audio (np.ndarray): Audio data for a single channel.
    sample_rate (int): Sample rate of the audio.
    channel_name (str): Name of the channel ('left' or 'right').
    
    Returns:
    Dict: Dictionary containing channel name and transcriptions.
    """
    try:
        transcriptions = inference.transcribe_batch(audio, sample_rate)
        return {
            "channel": channel_name,
            "transcriptions": transcriptions
        }
    except Exception as e:
        logging.error(f"Error transcribing {channel_name} channel: {str(e)}")
        return {
            "channel": channel_name,
            "transcriptions": [],
            "error": str(e)
        }


def process_audio_file(inference, file_path: str) -> Dict:
    """
    Process a single audio file: load, split channels, and transcribe.
    
    Args:
    inference: The inference object with a transcribe_batch method.
    file_path (str): Path to the audio file.
    
    Returns:
    Dict: Dictionary containing file name and channel transcriptions.
    """
    audio_result = load_audio(file_path)
    if audio_result is None:
        return {"file_name": os.path.basename(file_path), "error": "Failed to load audio"}
    
    left_channel, right_channel, sample_rate = audio_result
    
    return {
        "file_name": os.path.basename(file_path),
        "channels": [
            transcribe_channel(inference, left_channel, sample_rate, "left"),
            transcribe_channel(inference, right_channel, sample_rate, "right")
        ]
    }


def batch_inference(inference, 
                    audio_files: List[str], 
                    output_dir: str, 
                    file_name: str) -> None:
    """
    Perform batch inference on a list of audio files and save results to a JSON file.
    
    Args:
    inference: The inference object with a transcribe_batch method.
    audio_files (List[str]): List of paths to audio files.
    output_dir (str): Directory where the JSON file will be saved.
    """
    results = []
    for file in audio_files:
        logging.info(f"Processing file: {file}")
        result = process_audio_file(inference, file)
        results.append(result)
    
        output_file = os.path.join(output_dir, file_name)
        try:
            with open(output_file, 'a+', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            logging.info(f"Results written to {output_file}")
        except Exception as e:
            logging.error(f"Error writing results to JSON: {str(e)}")


def stream_inference(inference, audio_file):
    def audio_generator():
        audio, sample_rate = load_audio(audio_file)
        chunk_size = int(sample_rate * 1)  # 1 second chunks
        for i in range(0, len(audio), chunk_size):
            yield audio[i:i+chunk_size]

    print(f"Streaming transcription for file: {audio_file}")
    for transcription in inference.transcribe_stream(audio_generator()):
        print(f"Partial transcription: {transcription}")
    print()


def parse_args():
    parser = HfArgumentParser(InferenceArguments)
    args = parser.parse_args_into_dataclasses()[0]
    return args


def main():
    args = parse_args()
    inference = ChineseTaiwaneseASRInference(
        model_path=args.model_path,
        device=args.device,
        use_peft=args.use_peft,
        language=args.language,
        use_timestamps=args.use_timestamps
    )
    if args.mode == "batch":
        batch_inference(inference, 
                        args.audio_files,
                        args.output_dir,
                        args.file_name)
    else:
        for audio_file in args.audio_files:
            stream_inference(inference, audio_file)


if __name__ == "__main__":
    main()