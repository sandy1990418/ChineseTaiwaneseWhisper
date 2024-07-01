import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel, PeftConfig
import logging
from typing import Generator, List, Union, Tuple


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
            
            # Set the language token without using forced_decoder_ids
            self.language_token = self.processor.tokenizer.convert_tokens_to_ids(f"<|{language}|>")
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
            
            # Create attention mask
            attention_mask = torch.ones_like(input_features)
            attention_mask = attention_mask.to(self.device)
            
            generated_ids = self.model.generate(
                input_features,
                attention_mask=attention_mask,
                language=self.language,
                task="transcribe"
            )
            
            transcriptions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            return transcriptions
        except Exception as e:
            logger.error(f"Error in transcribe_batch: {e}")
            return [f"Error in transcription: {str(e)}"]

    @torch.no_grad()
    def transcribe_stream(self, audio_stream: Generator[np.ndarray, None, None], sample_rate: int = 16000, chunk_length_s: float = 1.0) -> Generator[str, None, None]:
        chunk_length = int(chunk_length_s * sample_rate)
        buffer = np.array([], dtype=np.float32)

        for chunk in audio_stream:
            buffer = np.concatenate([buffer, chunk])
            
            while len(buffer) >= chunk_length:
                audio_chunk = buffer[:chunk_length]
                buffer = buffer[chunk_length:]

                input_features = self.processor(audio_chunk, sampling_rate=sample_rate, return_tensors="pt").input_features
                input_features = input_features.to(self.device)

                # Generate transcription
                generated_ids = self.model.generate(
                    input_features, 
                    forced_decoder_ids=self.forced_decoder_ids,
                    language=self.language
                )
                transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                yield transcription.strip()

        # Process any remaining audio in the buffer
        if len(buffer) > 0:
            input_features = self.processor(buffer, sampling_rate=sample_rate, return_tensors="pt").input_features
            input_features = input_features.to(self.device)
            generated_ids = self.model.generate(
                input_features, 
                forced_decoder_ids=self.forced_decoder_ids,
                language=self.language
            )
            transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            yield transcription.strip()

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