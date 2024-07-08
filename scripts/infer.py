import argparse
import torch
import numpy as np
import soundfile as sf
from src.inference.flexible_inference import ChineseTaiwaneseASRInference


def load_audio(file_path):
    audio, sample_rate = sf.read(file_path)
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    return audio, sample_rate


def batch_inference(model, audio_files, use_timestamps):
    audio_batch = [load_audio(file)[0] for file in audio_files]
    transcriptions = model.transcribe_batch(audio_batch)
    for file, transcription in zip(audio_files, transcriptions):
        print(f"File: {file}")
        print(f"Transcription: {transcription}\n")


def stream_inference(model, audio_file, use_timestamps):
    def audio_generator():
        audio, sample_rate = load_audio(audio_file)
        chunk_size = int(sample_rate * 1)  # 1 second chunks
        for i in range(0, len(audio), chunk_size):
            yield audio[i:i+chunk_size]

    print(f"Streaming transcription for file: {audio_file}")
    for transcription in model.transcribe_stream(audio_generator()):
        print(f"Partial transcription: {transcription}")
    print()


def main():
    parser = argparse.ArgumentParser(description="ASR Inference Script")
    parser.add_argument("--model_path", required=True, help="Path to the ASR model")
    parser.add_argument("--audio_files", nargs="+", required=True, help="Path to audio file(s)")
    parser.add_argument("--mode", choices=["batch", "stream"], default="batch", help="Inference mode")
    parser.add_argument("--use_timestamps", action="store_true", help="Include timestamps in transcription")
    parser.add_argument("--use_peft", action="store_true", help="Use PEFT model")
    parser.add_argument("--language", default="chinese", help="Language of the audio (e.g., 'chinese', 'taiwanese')")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for inference")
    
    args = parser.parse_args()

    model = ChineseTaiwaneseASRInference(
        model_path=args.model_path,
        device=args.device,
        use_peft=args.use_peft,
        language=args.language,
        use_timestamps=args.use_timestamps
    )

    if args.mode == "batch":
        batch_inference(model, args.audio_files, args.use_timestamps)
    else:
        for audio_file in args.audio_files:
            stream_inference(model, audio_file, args.use_timestamps)


if __name__ == "__main__":
    main()