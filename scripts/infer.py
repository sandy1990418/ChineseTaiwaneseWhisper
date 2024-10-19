from transformers import HfArgumentParser
import numpy as np
from typing import Dict, List
import soundfile as sf
from src.inference.flexible_inference import ChineseTaiwaneseASRInference
from src.utils.logging import logger
import os
import json
from src.config import InferenceArguments
import tqdm


def get_wav_files(path: str) -> List[str]:
    """
    Get all WAV files from a directory or return the file if it's a WAV file.
    
    Args:
    path (str): Path to a directory or a file.
    
    Returns:
    List[str]: List of WAV file paths.
    """
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.lower().endswith('.wav')]
        return files
    elif os.path.isfile(path) and path.lower().endswith('.wav'):
        return [path]
    else:
        return []
    

def load_audio(file_path):
    audio, sample_rate = sf.read(file_path)
    # Convert to mono if stereo
    # Check if the audio is stereo
    if audio.ndim == 2 and audio.shape[1] == 2:
        return audio[:, 0], audio[:, 1], sample_rate
    else:
        return audio, sample_rate


def transcribe_channel(
    inference, audio: np.ndarray, sample_rate: int, channel_name: str
) -> Dict:
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
        return {"channel": channel_name, "transcriptions": transcriptions}
    except Exception as e:
        logger.error(f"Error transcribing {channel_name} channel: {str(e)}")
        return {"channel": channel_name, "transcriptions": [], "error": str(e)}


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
        return {
            "file_name": os.path.basename(file_path),
            "error": "Failed to load audio",
        }

    if len(audio_result) == 3:
        left_channel, right_channel, sample_rate = audio_result

        return {
            "file_name": os.path.basename(file_path),
            "channels": [
                transcribe_channel(inference, left_channel, sample_rate, "left"),
                transcribe_channel(inference, right_channel, sample_rate, "right"),
            ],
        }
    else:
        channel, sample_rate = audio_result

        return {
            "file_name": os.path.basename(file_path),
            "channels": [
                transcribe_channel(inference, channel, sample_rate, "single_channel"),
            ],
        }


def batch_inference(
    inference, audio_files: List[str], output_dir: str, file_name: str
) -> None:
    """
    Perform batch inference on a list of audio files and save results to a JSON file.

    Args:
    inference: The inference object with a transcribe_batch method.
    audio_files (List[str]): List of paths to audio files.
    output_dir (str): Directory where the JSON file will be saved.
    """
    results = []
    audio_files_dir = []
    for file in tqdm.tqdm(audio_files):
        if 'wav' not in file:
            audio_files = get_wav_files(file)
            audio_files_dir.extend(audio_files)
        else:
            audio_files_dir.append(file)

    for file in tqdm.tqdm(audio_files_dir):
        logger.info(f"Processing file: {file}")
        result = process_audio_file(inference, file)
        results.append(result)

    output_file = os.path.join(output_dir, file_name)
    try:
        with open(output_file, "a+", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        logger.info(f"Results written to {output_file}")
    except Exception as e:
        logger.error(f"Error writing results to JSON: {str(e)}")


def stream_inference(inference, audio_file):
    def audio_generator():
        audio, sample_rate = load_audio(audio_file)
        chunk_size = int(sample_rate * 1)  # 1 second chunks
        for i in range(0, len(audio), chunk_size):
            yield audio[i: i + chunk_size]

    logger.info(f"Streaming transcription for file: {audio_file}")
    for transcription in inference.transcribe_stream(audio_generator()):
        logger.info(f"Partial transcription: {transcription}")


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
        use_timestamps=args.use_timestamps,
    )
    if args.mode == "batch":
        # # Process each input path (file or directory)
        # all_audio_files = []
        # for path in args.audio_files:
        #     all_audio_files.extend(get_wav_files(path))
        # if not all_audio_files:
        #     logger.warning("No WAV files found in the specified paths.")
        #     return
        batch_inference(inference, args.audio_files, args.output_dir, args.file_name)
    else:
        for audio_file in args.audio_files:
            stream_inference(inference, audio_file)


if __name__ == "__main__":
    main()

# TODO : convert model to Ctranslate2
# TODO : Distil-model
# TODO : Teacher-student
# TODO : Speculative decoding
