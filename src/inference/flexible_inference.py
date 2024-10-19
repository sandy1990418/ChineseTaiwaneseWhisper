import torch
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
# from transformers import WhisperProcessor, WhisperForConditionalGeneration
from peft import PeftModel, PeftConfig
from src.utils.logging import logger
from typing import Generator, List
from collections import deque
import re
import tqdm
import librosa
import time
from concurrent.futures import ThreadPoolExecutor  # , as_completed


class ChineseTaiwaneseASRInference:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        use_peft: bool = False,
        language: str = "chinese",
        use_timestamps: bool = False,
        *args,
        **kwargs,
    ):

        self.device = device
        self.language = language
        self.use_timestamps = use_timestamps
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True
        )
        self.get_speech_timestamps, _, read_audio, _, _ = utils

        try:
            if use_peft:
                config = PeftConfig.from_pretrained(model_path)
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    config.base_model_name_or_path
                )
                self.model = PeftModel.from_pretrained(self.model, model_path)
            else:
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(model_path)

            self.model.to(device)
            # BUG FIX: https://medium.com/@bofenghuang7/what-i-learned-from-whisper-fine-tuning-event-2a68dab1862
            # included in the training
            self.model.config.forced_decoder_ids = None
            self.model.config.suppress_tokens = []
            # to use gradient checkpointing
            self.model.config.use_cache = False

            self.processor = AutoProcessor.from_pretrained(model_path)

            # Set the language token without using forced_decoder_ids
            self.language_token = self.processor.tokenizer.convert_tokens_to_ids(
                f"<|{language}|>"
            )
            self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language=language, task="transcribe"
            )

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
                inputs = self.processor(
                    chunk,
                    return_tensors="pt",
                    truncation=False,
                    return_attention_mask=True,
                    sampling_rate=16000,
                )

                input_features = inputs.input_features.to(self.device)
                if isinstance(self.language, type(None)):
                    # Detect Language
                    # https://discuss.huggingface.co/t/language-detection-with-whisper/26003/14
                    language_token = self.model.generate(input_features, max_new_tokens=1)[0, 1]
                    language_token = self.processor.tokenizer.decode(language_token)
                    language_token = re.search(r'<\|(.+?)\|>', language_token).group(1)
                else:
                    language_token = self.language
                
                logger.info(f"The Language Token is {language_token}")

                generated_ids = self.model.generate(
                    input_features,
                    language=language_token,  # self.language,
                    task="transcribe",
                    return_timestamps=self.use_timestamps,
                )
                chunk_trans = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    decode_with_timestamps=self.use_timestamps,
                )[0]
                if self.use_timestamps:
                    chunk_trans, cumulative_duration = self._process_timestamps(
                        chunk_trans,
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
    def transcribe_stream(
        self,
        audio_stream: Generator[np.ndarray, None, None],
        sample_rate: int = 16000,
        chunk_length_s: float = 30.0,
        stride_length_s: float = 2,
    ) -> Generator[dict, None, None]:
        chunk_length = int(chunk_length_s * sample_rate)
        stride_length = int(stride_length_s * sample_rate)
        audio_buffer = deque(maxlen=chunk_length)
        futures = []

        for chunk in audio_stream:
            audio_buffer.extend(chunk)

            while len(audio_buffer) >= chunk_length:
                # Extract the chunk
                audio_chunk = [audio_buffer.popleft() for _ in range(chunk_length)]
                audio_chunk = np.array(audio_chunk)

                # Re-insert the last 'stride_length' samples back into the buffer for overlap
                if stride_length > 0:
                    overlap_samples = audio_chunk[-stride_length:]
                    audio_buffer.extendleft(overlap_samples[::-1])

                # Submit the chunk for processing
                future = self.executor.submit(
                    self.process_chunk, audio_chunk, sample_rate
                )
                futures.append(future)

                # Break to allow processing and avoid high latency
                break

            # Process completed futures
            completed_futures = [f for f in futures if f.done()]
            for future in completed_futures:
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
        input_features = self.processor(
            audio_chunk, sampling_rate=sample_rate, return_tensors="pt"
        ).input_features

        input_features = input_features.to(self.device)
        language_token = self.model.generate(input_features, max_new_tokens=1)[0, 1]
        language_token = self.processor.tokenizer.decode(language_token)
        language_token = re.search(r'<\|(.+?)\|>', language_token).group(1)
        
        logger.info(f"The Language Token is {language_token}")

        generated_ids = self.model.generate(
            input_features,
            forced_decoder_ids=self.forced_decoder_ids,
            language=language_token,  # self.language,
            return_timestamps=self.use_timestamps,
            task="transcribe",
            num_beams=5,            # Use beam search with 5 beams
            early_stopping=True,  
        )

        # if isinstance(generated_ids, torch.Tensor):
        #     if generated_ids.dim() == 2:
        #         generated_ids = generated_ids.squeeze(0)
        #     elif generated_ids.dim() > 2:
        #         generated_ids = generated_ids.view(-1)
        # elif isinstance(generated_ids, list):
        #     generated_ids = torch.tensor(generated_ids).view(-1)

        # if isinstance(generated_ids, torch.Tensor):
        #     generated_ids = (
        #         generated_ids.squeeze(0)
        #         if generated_ids.dim() == 2
        #         else generated_ids.view(-1)
        #     )
        # elif isinstance(generated_ids, list):
        #     generated_ids = torch.tensor(generated_ids).view(-1)

        # if self.use_timestamps:  # self.processor.batch_decode(generated_ids,
        #     transcription = self.processor.decode(generated_ids,
        #                                           skip_special_tokens=True,
        #                                           decode_with_timestamps=self.use_timestamps)
        #     transcription = self._process_timestamps(transcription)
        # else:
        #     transcription = self.processor.decode(generated_ids,
        #                                           skip_special_tokens=True,
        #                                           decode_with_timestamps=self.use_timestamps)
        transcription = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            decode_with_timestamps=self.use_timestamps,
        )[0]
        if self.use_timestamps:
            transcription = self._process_timestamps(transcription)[0]

        end_time = time.time()
        processing_time = end_time - start_time
        speed = (
            len(audio_chunk) / sample_rate / processing_time
            if processing_time > 0
            else 0
        )

        return {"transcription": transcription.strip(), "speed": speed}

    def is_speech(self, audio_chunk: np.ndarray, sample_rate: int) -> bool:
        audio_tensor = torch.FloatTensor(audio_chunk)
        # Adjusted VAD parameters for better accuracy
        speech_timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.vad_model,
            sampling_rate=sample_rate,
            threshold=0.5,  # Adjusted threshold
            min_speech_duration_ms=250,  # Minimum speech duration
            min_silence_duration_ms=100,  # Minimum silence duration
        )
        return len(speech_timestamps) > 0

    def _process_audio(
        self,
        audio,
        sample_rate: int = 16000,
        target_sample_rate: int = 16000,
        chunk_length: int = 420000,
    ) -> List[np.ndarray]:
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
            audio = librosa.resample(
                audio, orig_sr=sample_rate, target_sr=target_sample_rate
            )

        # Normalize audio
        audio_array = audio / np.max(np.abs(audio))

        return [
            audio_array[i: i + chunk_length]
            for i in range(0, audio_array.shape[0], chunk_length)
        ]

    def _process_timestamps(self, transcription: str, offset: float = 0) -> str:
        """Process transcription with timestamps."""
        # Regular expression to match timestamp tokens
        pattern = r"<\|(\d+\.\d+)\|>"

        # Split the transcription into segments based on timestamp tokens
        segments = re.split(pattern, transcription)
        segments = list(filter(None, segments))

        # Process segments and timestamps
        formatted_transcription = []

        for i in range(0, len(segments) - 2, 3):
            timestamp_start = float(segments[i]) + offset
            text = segments[i + 1].strip()
            try:
                timestamp_end = float(segments[i + 2]) + offset
            except ValueError:
                timestamp_end = offset + 30

            if text:  # Only add non-empty segments
                text = self.remove_duplicates(text)
                formatted_transcription.extend(
                    [f"[{timestamp_start:.2f}]-[{timestamp_end:.2f}]{text}"]
                )
            offset = timestamp_end
        return formatted_transcription, offset

    def remove_duplicates(self, input_str):
        # Split the input string into individual phrases
        phrases = input_str.split(",")

        # Remove duplicates while maintaining order
        unique_phrases = list(dict.fromkeys(phrases))

        # Join the unique phrases back into a single string
        return ",".join(unique_phrases)
