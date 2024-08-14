import torch
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration, LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel, PeftConfig
import logging
from typing import Generator, List
from collections import deque
import re
import tqdm
import librosa
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChineseTaiwaneseASRInference:
    def __init__(self, 
                 model_path: str, 
                 device: str = "cuda", 
                 use_peft: bool = False, 
                 language: str = "chinese",
                 use_timestamps: bool = False, 
                 *args, 
                 **kwargs):

        self.device = device
        self.language = language
        self.use_timestamps = use_timestamps
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                               model='silero_vad',
                                               force_reload=True)
        self.get_speech_timestamps, _, read_audio, _, _ = utils

        try:
            if use_peft:
                config = PeftConfig.from_pretrained(model_path)
                self.model = WhisperForConditionalGeneration.from_pretrained(config.base_model_name_or_path)
                self.model = PeftModel.from_pretrained(self.model, model_path)
            else:
                self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
            
            self.model.to(device)
            # BUG FIX: https://medium.com/@bofenghuang7/what-i-learned-from-whisper-fine-tuning-event-2a68dab1862
            # included in the training
            self.model.config.forced_decoder_ids = None
            self.model.config.suppress_tokens = []
            # to use gradient checkpointing
            self.model.config.use_cache = False

            self.processor = WhisperProcessor.from_pretrained(model_path)
            
            # Set the language token without using forced_decoder_ids
            self.language_token = self.processor.tokenizer.convert_tokens_to_ids(f"<|{language}|>")
            self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task="transcribe")

        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    @torch.no_grad()
    def transcribe_batch(self, audio, sample_rate):
        try:
            transcriptions = []
            if audio is None:
                transcriptions.append("No valid audio input provided.")
            audio_chunks = self._process_audio(audio, sample_rate)
            chunk_transcriptions = []
            cumulative_duration = 0
            for chunk in tqdm.tqdm(audio_chunks):
                inputs = self.processor(chunk, 
                                        return_tensors="pt", 
                                        truncation=False, 
                                        return_attention_mask=True, 
                                        sampling_rate=16000)

                input_features = inputs.input_features.to(self.device)
                generated_ids = self.model.generate(
                    input_features,
                    language=self.language,
                    task="transcribe",
                    return_timestamps=self.use_timestamps
                )
                chunk_trans = self.processor.batch_decode(generated_ids, 
                                                          skip_special_tokens=True,
                                                          decode_with_timestamps=self.use_timestamps)[0]
                if self.use_timestamps:
                    chunk_trans, cumulative_duration = self._process_timestamps(chunk_trans, 
                                                                                cumulative_duration,
                                                                                )
                chunk_transcriptions.extend(chunk_trans)

            full_transcription = "".join(chunk_transcriptions)
            transcriptions.append(full_transcription)
            
            return transcriptions
        except Exception as e:
            logger.error(f"Error in transcribe_batch: {e}")
            return [f"Error in transcription: {str(e)}"]

    @torch.no_grad()
    def transcribe_stream(self, 
                          audio_stream: Generator[np.ndarray, None, None], 
                          sample_rate: int = 16000, 
                          chunk_length_s: float = 30.0, 
                          stride_length_s: float = 2) -> Generator[dict, None, None]:
        chunk_length = int(chunk_length_s * sample_rate)
        stride_length = int(stride_length_s * sample_rate)
        audio_buffer = deque(maxlen=chunk_length)
        futures = []

        for chunk in audio_stream:
            audio_buffer.extend(chunk)

            if audio_buffer:  # len(audio_buffer) >= chunk_length:
                audio_chunk = np.array(audio_buffer)
                future = self.executor.submit(self.process_chunk, audio_chunk, sample_rate)
                futures.append(future)

                # Remove strided part from the beginning of the buffer
                for _ in range(stride_length):
                    if audio_buffer:
                        audio_buffer.popleft()

            # Process completed futures
            for future in as_completed(futures):
                result = future.result()
                if result:
                    yield result
                futures.remove(future)

        # Process any remaining audio in the buffer
        if audio_buffer:
            remaining_audio = np.array(audio_buffer)
            result = self.process_chunk(remaining_audio, sample_rate)
            if result:
                yield result

    def process_chunk(self, audio_chunk: np.ndarray, sample_rate: int) -> dict:
        start_time = time.time()

        # Check if the audio chunk contains speech
        if not self.is_speech(audio_chunk, sample_rate):
            return None

        # Preprocess audio
        audio_chunk = librosa.util.normalize(audio_chunk)
        
        # Process audio chunk
        input_features = self.processor(audio_chunk, 
                                        sampling_rate=sample_rate, 
                                        return_tensors="pt").input_features
        input_features = input_features.to(self.device)

        generated_ids = self.model.generate(
            input_features, 
            forced_decoder_ids=self.forced_decoder_ids,
            language=self.language,
            return_timestamps=self.use_timestamps
        )

        # if isinstance(generated_ids, torch.Tensor):
        #     if generated_ids.dim() == 2:
        #         generated_ids = generated_ids.squeeze(0)
        #     elif generated_ids.dim() > 2:
        #         generated_ids = generated_ids.view(-1)
        # elif isinstance(generated_ids, list):
        #     generated_ids = torch.tensor(generated_ids).view(-1)

        if isinstance(generated_ids, torch.Tensor):
            generated_ids = generated_ids.squeeze(0) if generated_ids.dim() == 2 else generated_ids.view(-1)
        elif isinstance(generated_ids, list):
            generated_ids = torch.tensor(generated_ids).view(-1)

        # if self.use_timestamps:  # self.processor.batch_decode(generated_ids, 
        #     transcription = self.processor.decode(generated_ids, 
        #                                           skip_special_tokens=True,
        #                                           decode_with_timestamps=self.use_timestamps)
        #     transcription = self._process_timestamps(transcription)
        # else:
        #     transcription = self.processor.decode(generated_ids, 
        #                                           skip_special_tokens=True,
        #                                           decode_with_timestamps=self.use_timestamps)
        transcription = self.processor.decode(generated_ids, 
                                              skip_special_tokens=True,
                                              decode_with_timestamps=self.use_timestamps)        
        if self.use_timestamps:  
            transcription = self._process_timestamps(transcription) 
               
        end_time = time.time()
        processing_time = end_time - start_time
        speed = len(audio_chunk) / sample_rate / processing_time if processing_time > 0 else 0

        return {"transcription": transcription.strip(), "speed": speed}

    def is_speech(self, audio_chunk: np.ndarray, sample_rate: int) -> bool:
        audio_tensor = torch.FloatTensor(audio_chunk)
        speech_timestamps = self.get_speech_timestamps(audio_tensor, self.vad_model, sampling_rate=sample_rate)
        return len(speech_timestamps) > 0

    def _process_audio(self, 
                       audio, 
                       sample_rate: int = 16000,
                       target_sample_rate: int = 16000,
                       chunk_length: int = 480000) -> List[np.ndarray]:
        """Process and pad or trim the audio array to chunk_length."""
        """Split audio into chunks of 30 seconds (480000 samples at 16kHz)."""
        if isinstance(audio, np.ndarray):
            audio_array = audio
        elif isinstance(audio, tuple):
            audio_array = audio[1] if len(audio) > 1 else audio[0]
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio)}")
        
        # Resample to 16kHz if necessary
        if sample_rate != 16000:
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
                
        # Normalize audio
        audio_array = audio / np.max(np.abs(audio))

        return [audio_array[i:i+chunk_length] for i in range(0, audio_array.shape[0], chunk_length)]

    def _process_timestamps(self, transcription: str, offset: float = 0) -> str:
        """Process transcription with timestamps."""
        # Regular expression to match timestamp tokens
        pattern = r'<\|(\d+\.\d+)\|>'
        
        # Split the transcription into segments based on timestamp tokens
        segments = re.split(pattern, transcription)
        segments = list(filter(None, segments))
        
        # Process segments and timestamps
        formatted_transcription = []

        for i in range(0, len(segments) - 2, 3):
            timestamp_start = float(segments[i]) + offset
            text = segments[i + 1].strip()
            try:
                timestamp_end = float(segments[i+2]) + offset
            except ValueError:
                timestamp_end = offset+30

            if text:  # Only add non-empty segments
                text = self.remove_duplicates(text)
                formatted_transcription.extend([f"[{timestamp_start:.2f}]-[{timestamp_end:.2f}]{text}"])
            offset = timestamp_end
        return formatted_transcription, offset

    def remove_duplicates(self, input_str):
        # Split the input string into individual phrases
        phrases = input_str.split(',')
        
        # Remove duplicates while maintaining order
        unique_phrases = list(dict.fromkeys(phrases))
        
        # Join the unique phrases back into a single string
        return ','.join(unique_phrases)


class FusionWhisperLLaMAInference(ChineseTaiwaneseASRInference):
    def __init__(self, 
                 whisper_model_path: str,
                 llama_model_path: str,
                 device: str = "cuda",
                 use_peft: bool = False,
                 language: str = "chinese",
                 use_timestamps: bool = False,
                 lm_weight: float = 0.1,
                 *args, 
                 **kwargs):
        super().__init__(whisper_model_path, device, use_peft, language, use_timestamps, *args, **kwargs)
        
        # Initialize LLaMA
        try:
            if not isinstance(llama_model_path, str):
                raise ValueError(f"llama_model_path must be a string, got {type(llama_model_path)}")
            
            self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model_path)
            self.llama_model = LlamaForCausalLM.from_pretrained(llama_model_path).to(device)
            logger.info(f"LLaMA model loaded successfully from {llama_model_path}")
        except Exception as e:
            logger.error(f"Error loading LLaMA model: {str(e)}")
            raise

        self.lm_weight = lm_weight

    def align_token_spaces(self, whisper_tokens: List[int]) -> List[int]:
        """Align token spaces between Whisper and LLaMA"""
        whisper_text = self.processor.decode(whisper_tokens)
        return self.llama_tokenizer.encode(whisper_text, add_special_tokens=False)

    def get_llm_log_probs(self, aligned_tokens: List[int]) -> torch.Tensor:
        """Get log probabilities from LLaMA for error correction"""
        input_ids = torch.tensor(aligned_tokens).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.llama_model(input_ids)
            log_probs = torch.log_softmax(outputs.logits, dim=-1)
        return log_probs

    def optimized_decoding(self, audio_features: torch.Tensor, max_length: int = 200) -> torch.Tensor:
        """Implement optimized decoding strategy with shallow fusion"""
        input_ids = torch.tensor([[self.model.config.decoder_start_token_id]]).to(self.device)
        
        for _ in range(max_length):
            with torch.no_grad():
                outputs = self.model(
                    input_features=audio_features,
                    decoder_input_ids=input_ids
                )
            whisper_logits = outputs.logits[:, -1, :]
            whisper_log_probs = torch.log_softmax(whisper_logits, dim=-1)
            
            aligned_tokens = self.align_token_spaces(input_ids[0].tolist())
            llm_log_probs = self.get_llm_log_probs(aligned_tokens)[:, -1, :]
            
            combined_log_probs = whisper_log_probs + self.lm_weight * llm_log_probs
            
            next_token = torch.argmax(combined_log_probs, dim=-1).unsqueeze(0)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            if next_token.item() == self.processor.tokenizer.eos_token_id:
                break
        
        return input_ids

    @torch.no_grad()
    def transcribe_batch(self, audio, sample_rate):
        try:
            transcriptions = []
            if audio is None:
                transcriptions.append("No valid audio input provided.")
            audio_chunks = self._process_audio(audio, sample_rate)
            chunk_transcriptions = []
            cumulative_duration = 0
            for chunk in tqdm.tqdm(audio_chunks):
                inputs = self.processor(chunk, 
                                        return_tensors="pt", 
                                        truncation=False, 
                                        return_attention_mask=True, 
                                        sampling_rate=16000)

                input_features = inputs.input_features.to(self.device)
                generated_ids = self.optimized_decoding(input_features)
                chunk_trans = self.processor.decode(generated_ids[0], 
                                                    skip_special_tokens=True,
                                                    decode_with_timestamps=self.use_timestamps)
                if self.use_timestamps:
                    chunk_trans, cumulative_duration = self._process_timestamps(chunk_trans, cumulative_duration)
                chunk_transcriptions.extend(chunk_trans)

            full_transcription = "".join(chunk_transcriptions)
            transcriptions.append(full_transcription)
            
            return transcriptions
        except Exception as e:
            logger.error(f"Error in transcribe_batch: {e}")
            return [f"Error in transcription: {str(e)}"]

    def process_chunk(self, audio_chunk: np.ndarray, sample_rate: int) -> dict:
        start_time = time.time()

        if not self.is_speech(audio_chunk, sample_rate):
            return None

        audio_chunk = librosa.util.normalize(audio_chunk)
        
        input_features = self.processor(audio_chunk, 
                                        sampling_rate=sample_rate, 
                                        return_tensors="pt").input_features
        input_features = input_features.to(self.device)

        generated_ids = self.optimized_decoding(input_features)

        transcription = self.processor.decode(generated_ids[0], 
                                              skip_special_tokens=True,
                                              decode_with_timestamps=self.use_timestamps)        
        if self.use_timestamps:  
            transcription = self._process_timestamps(transcription)[0]  # Only take the formatted transcription
               
        end_time = time.time()
        processing_time = end_time - start_time
        speed = len(audio_chunk) / sample_rate / processing_time if processing_time > 0 else 0

        return {"transcription": transcription.strip(), "speed": speed}