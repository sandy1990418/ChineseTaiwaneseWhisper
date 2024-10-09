from collections import defaultdict
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoModelForCausalLM,
)
import torch
from transformers import LlamaTokenizerFast, WhisperTokenizer
from src.utils.logging import logger

# TODO: A lot of bugggg!
# https://github.com/mtkresearch/generative-fusion-decoding

class ByteTokenizer:
    def tokenize_from_byte(self, byte_str):
        str_part = byte_str.decode("utf8", errors="ignore")
        return self(str_part, add_special_tokens=False).input_ids

    def convert_ids_to_bytes(self, ids):
        raise NotImplementedError

    def get_matched_ids_from_prefix(self, byte_prefix):
        if not hasattr(self, "_prefix_to_ids"):
            self._prefix_to_ids = defaultdict(list)
            for i in range(self.vocab_size):
                b = self.convert_ids_to_bytes(i)
                for j in range(1, len(b) + 1):
                    prefix = b[:j]
                    self._prefix_to_ids[prefix].append(i)
        return self._prefix_to_ids.get(byte_prefix, [])

    def get_alternative_ids(self, seq_ids):
        alternative_ids = [None] * len(seq_ids)
        prefix_from_last = b""
        pointer_from_last = 1
        while pointer_from_last <= len(seq_ids):
            id_to_convert = seq_ids[-pointer_from_last]
            converted_bytes = self.convert_ids_to_bytes(id_to_convert)
            prefix_from_last = converted_bytes + prefix_from_last
            alternative_ids[-pointer_from_last] = self.get_matched_ids_from_prefix(
                prefix_from_last
            )
            pointer_from_last += 1

        return alternative_ids


class LlamaByteTokenizer(LlamaTokenizerFast, ByteTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bytetokens_to_ids = {}
        for s, i in self.get_vocab().items():
            b = self._convert_token_to_byte(s)
            if b in self.bytetokens_to_ids:
                if self.bytetokens_to_ids[b] < i:
                    self.bytetokens_to_ids[b] = i
            else:
                self.bytetokens_to_ids[b] = i

    def convert_ids_to_bytes(self, ids):
        if isinstance(ids, int):
            tokens = self.convert_ids_to_tokens(ids, skip_special_tokens=False)
            return self._convert_token_to_byte(tokens)
        else:
            tokens = self.convert_ids_to_tokens(ids, skip_special_tokens=False)
            if isinstance(tokens, str):
                return self._convert_token_to_byte(tokens)
            return b"".join([self._convert_token_to_byte(t) for t in tokens])

    def _convert_token_to_byte(self, token):
        SPIECE_UNDERLINE = "▁"
        if token.startswith(SPIECE_UNDERLINE) and len(token) > 1:
            token = " " + token.lstrip(SPIECE_UNDERLINE)

        if token.startswith("<0x"):  # '<0xAB>' -> 'AB' -> b'\xAB'
            bs = bytes.fromhex(f"{token[3:5]}")
        else:
            bs = token.encode("utf8")
        return bs

    def tokenize_from_byte(self, byte_str):
        str_part = byte_str.decode("utf8", errors="ignore")
        encoded_str_part = str_part.encode("utf8")

        str_part_tokenized = self(str_part, add_special_tokens=False).input_ids
        leftover_string = byte_str[len(encoded_str_part):]
        for byte_int in leftover_string:
            byte_character = bytes([byte_int])
            str_part_tokenized.append(self.bytetokens_to_ids[byte_character])

        return str_part_tokenized


class WhisperByteTokenizer(WhisperTokenizer, ByteTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def convert_ids_to_bytes(self, ids, skip_special_tokens=True):
        if isinstance(ids, int):
            token = self.convert_ids_to_tokens(
                ids, skip_special_tokens=skip_special_tokens
            )
            return bytes([self.byte_decoder[c] for c in token])
        else:
            tokens = self.convert_ids_to_tokens(
                ids, skip_special_tokens=skip_special_tokens
            )
            return b"".join([bytes([self.byte_decoder[c] for c in s]) for s in tokens])


class GenerativeFusionDecoding:
    def __init__(
        self,
        asr_model: AutoModelForSpeechSeq2Seq,
        lm_model: AutoModelForCausalLM,
        asr_processor: AutoProcessor,
        lm_tokenizer: LlamaByteTokenizer,
        r: float = 0.2,
    ):
        self.asr_model = asr_model
        self.lm_model = lm_model
        self.asr_processor = asr_processor
        self.lm_tokenizer = lm_tokenizer
        self.r = r  # fusion weight

    def byte_level_probability(
        self, model, tokenizer, input_ids_or_features, generated_ids, past_key_values, is_asr_model=False
    ):
        input_ids = torch.tensor([generated_ids], device=model.device)

        if is_asr_model:
            # For ASR model (Whisper)
            outputs = model(
                input_features=input_ids_or_features,
                decoder_input_ids=input_ids,
                past_key_values=past_key_values,
            )
        else:
            # For LM model (LlamaForCausalLM)
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
            )

        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)

        # Use ByteTokenizer methods
        main_bytes = tokenizer.convert_ids_to_bytes(generated_ids)
        main_prob = torch.prod(probs[0, input_ids[0]]).item()

        # Get alternative token IDs that match the byte prefix
        alternative_ids = tokenizer.get_alternative_ids(generated_ids)

        alt_probs = 0
        for alt_id_list in alternative_ids:
            if alt_id_list:
                for alt_id in alt_id_list:
                    alt_prob = probs[0, alt_id].item()
                    alt_probs += alt_prob

        return main_prob + alt_probs

    def fuse_probabilities(self, asr_prob, lm_prob, t, k):
        if t < k:
            # At the beginning of the sequence, only use ASR model probabilities
            return asr_prob
        else:
            # Use fused probabilities
            return (1 - self.r) * asr_prob + self.r * lm_prob

    def decode(
        self,
        audio_input,
        sampling_rate=16000,
        prompt=None,
        beam_size=5,
        max_length=200,
        k=5,
        language="zh",
    ):
        device = next(self.asr_model.parameters()).device
        asr_inputs = self.asr_processor(
            audio_input, sampling_rate=sampling_rate, return_tensors="pt"
        ).to(device)

        # Initialize decoder input IDs
        decoder_start_token_id = self.asr_model.config.decoder_start_token_id
        decoder_input_ids = torch.tensor([[decoder_start_token_id]], device=device)

        # Get language and task-specific decoder prompt IDs
        forced_decoder_ids = self.asr_processor.get_decoder_prompt_ids(
            language=language, task="transcribe"
        )
        for _, token_id in forced_decoder_ids:
            decoder_input_ids = torch.cat(
                [decoder_input_ids, torch.tensor([[token_id]], device=device)], dim=1
            )

        beams = [
            {
                "sequence": decoder_input_ids.squeeze(0),
                "score": 0.0,
                "asr_state": None,
                "lm_state": None,
            }
        ]

        for t in range(max_length):
            candidates = []
            for beam in beams:
                # ASR model generation
                asr_output = self.asr_model.generate(
                    **asr_inputs,
                    decoder_input_ids=beam["sequence"].unsqueeze(0),
                    output_scores=True,
                    return_dict_in_generate=True,
                    max_new_tokens=1,
                    past_key_values=beam["asr_state"],
                    use_cache=True,
                )

                asr_token = asr_output.sequences[0, -1]

                asr_prob = self.byte_level_probability(
                    self.asr_model,
                    self.asr_processor.tokenizer,
                    asr_inputs.input_features,
                    beam["sequence"].tolist() + [asr_token.item()],
                    asr_output.past_key_values,
                    is_asr_model=True,
                )

                # LM model generation
                if prompt:
                    lm_input_ids = self.lm_tokenizer.tokenize_from_byte(prompt.encode('utf-8'))
                else:
                    lm_input_ids = []

                # Append the beam sequence
                lm_input_ids += beam["sequence"].tolist()

                # Convert to tensor
                if len(lm_input_ids) == 0:
                    lm_input = torch.tensor(
                        [[self.lm_model.config.bos_token_id]], device=device
                    )
                else:
                    lm_input = torch.tensor([lm_input_ids], device=device)

                # Create attention_mask
                attention_mask = torch.ones_like(lm_input)

                # Set pad_token_id
                pad_token_id = self.lm_model.config.pad_token_id
                if pad_token_id is None:
                    pad_token_id = self.lm_model.config.eos_token_id

                lm_output = self.lm_model.generate(
                    input_ids=lm_input,
                    attention_mask=attention_mask,
                    pad_token_id=pad_token_id,
                    max_new_tokens=1,
                    output_scores=True,
                    return_dict_in_generate=True,
                    past_key_values=beam["lm_state"],
                    use_cache=True,
                )

                lm_token = lm_output.sequences[0, -1]

                # Prepare LM input IDs for probability computation
                lm_input_ids += [lm_token.item()]

                lm_prob = self.byte_level_probability(
                    self.lm_model,
                    self.lm_tokenizer,
                    lm_input_ids,
                    lm_input_ids,
                    lm_output.past_key_values,
                    is_asr_model=False,
                )

                # Fuse probabilities
                fused_score = self.fuse_probabilities(asr_prob, lm_prob, t, k)

                new_sequence = torch.cat([beam["sequence"], asr_token.unsqueeze(0)])
                breakpoint()
                candidates.append(
                    {
                        "sequence": new_sequence,
                        "score": beam["score"] + fused_score,
                        "asr_state": asr_output.past_key_values,
                        "lm_state": lm_output.past_key_values,
                    }
                )

            # Select the best beam_size candidates
            beams = sorted(candidates, key=lambda x: x["score"], reverse=True)[:beam_size]

        # Return the highest scoring sequence
        return beams[0]["sequence"]



if __name__ == "__main__":
    # Load models and processors
    asr_model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-small")
    asr_processor = AutoProcessor.from_pretrained("openai/whisper-small")
    asr_processor.tokenizer = WhisperByteTokenizer.from_pretrained(
        "openai/whisper-small"
    )

    lm_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
    lm_tokenizer = LlamaByteTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    # Initialize GFD
    gfd = GenerativeFusionDecoding(asr_model, lm_model, asr_processor, lm_tokenizer)

    # Perform decoding
    audio_input = (
        "youtube_data/test/split_audio/test_0000_0000_0000.wav"  # Load your audio input
    )
    import librosa

    audio, _ = librosa.load(audio_input, sr=16000)
    result = gfd.decode(audio)

    # Output the result
    print(asr_processor.decode(result))




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