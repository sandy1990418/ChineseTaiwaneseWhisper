import argparse
import sounddevice as sd
import numpy as np
from src.inference.streaming import StreamingASRInference

def get_audio_stream(sample_rate=16000, chunk_duration=1):
    def audio_stream():
        with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32') as stream:
            while True:
                audio_chunk, _ = stream.read(int(sample_rate * chunk_duration))
                yield audio_chunk.flatten()

    return audio_stream()

def main():
    parser = argparse.ArgumentParser(description="Run streaming inference with Whisper")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned Whisper model")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (cuda or cpu)")
    args = parser.parse_args()

    inference = StreamingASRInference(args.model_path, device=args.device)
    audio_stream = get_audio_stream()

    print("Starting streaming inference. Speak into your microphone...")
    try:
        for transcription in inference.transcribe_stream(audio_stream):
            print(f"Transcription: {transcription}")
    except KeyboardInterrupt:
        print("Inference stopped by user.")

if __name__ == "__main__":
    main()