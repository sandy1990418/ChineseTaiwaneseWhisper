import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel, PeftConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChineseTaiwaneseASRInference:
    def __init__(self, model_path: str, device: str = "cuda", use_peft: bool = False, language: str = "chinese"):
        self.device = device
        self.language = language

        try:
            if use_peft:
                config = PeftConfig.from_pretrained(model_path)
                self.model = WhisperForConditionalGeneration.from_pretrained(config.base_model_name_or_path)
                self.model = PeftModel.from_pretrained(self.model, model_path)
            else:
                self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
            
            self.model.to(device)
            self.processor = WhisperProcessor.from_pretrained(model_path)
            
            # Set the language token
            self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task="transcribe")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    @torch.no_grad()
    def transcribe_batch(self, audio_batch):
        try:
            # Ensure minimum audio length and handle None inputs
            audio_batch = [self._process_audio(audio) for audio in audio_batch if audio is not None]
            
            if not audio_batch:
                return ["No valid audio input provided."]
            
            inputs = self.processor(audio_batch, return_tensors="pt", padding=True, sampling_rate=16000)
            input_features = inputs.input_features.to(self.device)
            
            generated_ids = self.model.generate(
                input_features,
                forced_decoder_ids=self.forced_decoder_ids,
                language=self.language
            )
            
            transcriptions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            return transcriptions
        except Exception as e:
            logger.error(f"Error in transcribe_batch: {e}")
            return [f"Error in transcription: {str(e)}"]

    @torch.no_grad()
    def transcribe_stream(self, audio_data, chunk_length_s: float = 30.0):
        try:
            audio_data = self._process_audio(audio_data)
            sample_rate = 16000  # Whisper expects 16kHz audio
            chunk_length = int(chunk_length_s * sample_rate)
            
            for i in range(0, len(audio_data), chunk_length):
                chunk = audio_data[i:i+chunk_length]
                if len(chunk) < chunk_length:
                    chunk = np.pad(chunk, (0, chunk_length - len(chunk)), 'constant')
                
                inputs = self.processor(chunk, return_tensors="pt", sampling_rate=sample_rate)
                input_features = inputs.input_features.to(self.device)
                
                generated_ids = self.model.generate(
                    input_features,
                    forced_decoder_ids=self.forced_decoder_ids,
                    language=self.language
                )
                
                chunk_transcription = self.processor.decode(generated_ids[0], skip_special_tokens=True)
                yield chunk_transcription
        except Exception as e:
            logger.error(f"Error in transcribe_stream: {e}")
            yield f"Error in streaming transcription: {str(e)}"

    def _process_audio(self, audio, target_length: int = 480000):
        """Process and pad or trim the audio array to target_length."""
        if isinstance(audio, np.ndarray):
            audio_array = audio
        elif isinstance(audio, tuple):
            audio_array = audio[1] if len(audio) > 1 else audio[0]
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio)}")

        audio_array = np.array(audio_array).flatten().astype(np.float32)
        
        # Normalize audio
        audio_array = audio_array / np.max(np.abs(audio_array))
        
        if len(audio_array) < target_length:
            return np.pad(audio_array, (0, target_length - len(audio_array)), 'constant')
        return audio_array[:target_length]