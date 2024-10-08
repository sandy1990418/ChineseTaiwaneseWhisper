from inference import FusionWhisperLLaMAInference
from src.utils.logging import logger
import librosa


def main():
    # Initialize the fusion model
    logger.info("Initializing FusionWhisperLLaMAInference model...")
    fusion_model = FusionWhisperLLaMAInference(
        whisper_model_path="openai/whisper-small",
        llama_model_path="taide/Llama3-TAIDE-LX-8B-Chat-Alpha1",
        device="cuda",
        use_peft=False,
        language="chinese",
        use_timestamps=True,
        lm_weight=0.1,
    )

    # Single audio file transcription
    audio_path = "S00001.wav"
    logger.info(f"Transcribing single audio file: {audio_path}")
    audio, sr = librosa.load(audio_path, sr=None)
    transcription = fusion_model.transcribe_batch(audio, sr)
    logger.info(f"Transcription: {transcription}")

    # Streaming transcription example
    logger.info("Starting streaming transcription...")

    def audio_stream_generator(
        audio_path, chunk_size=1600
    ):  # 0.1 second chunks at 16kHz
        audio, sr = librosa.load(audio_path, sr=16000)
        for i in range(0, len(audio), chunk_size):
            yield audio[i: i + chunk_size]

    for result in fusion_model.transcribe_stream(audio_stream_generator(audio_path)):
        if result:
            logger.info(f"Partial transcription: {result['transcription']}")
            logger.info(f"Speed: {result['speed']:.2f}x real-time")


if __name__ == "__main__":
    main()
