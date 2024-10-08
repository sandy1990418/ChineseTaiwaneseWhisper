import gradio as gr
import numpy as np
import torch
from src.inference.flexible_inference import ChineseTaiwaneseASRInference
from scipy import signal
import os
from datetime import datetime
import json
# import time
from transformers import HfArgumentParser
from src.config import GradioArguments
from typing import Optional, Union, Any 
from src.utils.logging import logger
from summary import summarize_transcript, process_transcripts 


class ASRProcessor:
    def __init__(
        self,
        language: str,
        model_choice: Optional[str] = "OpenAI Whisper Small",
        use_peft: Optional[bool] = False,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.language = language
        self.initialize_model(model_choice, use_peft)

    def initialize_model(self, model_choice, use_peft):
        if model_choice == "Custom (Finetuned)":
            model_path = "./whisper-finetuned-zh-tw"
        elif model_choice == "Custom (PEFT)":
            model_path = "./whisper-peft-finetuned-zh-tw"
        else:
            model_path = "openai/whisper-small"
        
        logger.info(f"Right now use model : {model_choice}")

        self.model = ChineseTaiwaneseASRInference(
            model_path, device=self.device, use_peft=use_peft, language=self.language
        )

    def reset_model(self):
        if self.model:
            del self.model
            torch.cuda.empty_cache()
        self.model = None


def log_to_json(
    message: str, cache_dir: str, cache_file_name: str, channel: Optional[str] = None
):
    cache_dir = os.path.join(os.getcwd(), cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    log_file = os.path.join(cache_dir, cache_file_name)

    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": message,
    }

    if not isinstance(channel, type(None)):
        log_entry.update(
            {
                "channel": channel,
            }
        )

    if os.path.exists(log_file):
        with open(log_file, "r+") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []

            logs.append(log_entry)

            f.seek(0)
            json.dump(logs, f, ensure_ascii=False, indent=4)
            f.truncate()
    else:
        with open(log_file, "w") as f:
            json.dump([log_entry], f, ensure_ascii=False, indent=4)


def convert_audio_sampling(audio):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    y = resample_audio(y, sr, 16000)
    sr = 16000
    if y.ndim == 2 and y.shape[1] == 2:
        if (np.around(y[:, 0]) == np.around(y[:, 1])).all():
            return y[:, 0], sr
        else:
            return y[:, 0], y[:, 1], sr
    return y, sr


def resample_audio(y, orig_sr, target_sr):
    if orig_sr != target_sr:
        num_samples = int(len(y) * float(target_sr) / orig_sr)
        y = signal.resample(y, num_samples)
    return y


def transcribe_batch(
    audio: Any,
    asr_processor: object,
    cache_dir: Union[str],
    cache_batch_filename: str
) -> str:
    """
    Transcribe audio input using the provided ASR processor.

    Args:
        audio: Audio input (file path, bytes, or None)
        asr_processor: ASR processor object
        cache_dir: Directory to store cache files
        cache_batch_filename: Name of the cache file

    Returns:
        str: Transcription result
    """
    logger.info(f"Starting transcription for {cache_batch_filename}")

    if audio is None:
        logger.warning("No audio input provided.")
        return "No audio input provided."
    try:
        audio_result = convert_audio_sampling(audio)
    except Exception as e:
        logger.error(f"Error converting audio: {str(e)}")
        return f"Error processing audio: {str(e)}"

    channels = ['left', 'right']
    transcriptions = []

    if len(audio_result) == 3:  # Stereo audio
        for idx, channel_audio in enumerate(audio_result[:2]):
            channel_transcription = _process_channel(
                channel_audio, 
                audio_result[-1], 
                asr_processor, 
                cache_dir, 
                cache_batch_filename, 
                channels[idx]
            )
            transcriptions.append(f"{channels[idx]}:{channel_transcription}\n")
    else:  # Mono audio
        mono_transcription = _process_channel(
            audio_result[0], 
            audio_result[-1], 
            asr_processor, 
            cache_dir, 
            cache_batch_filename
        )
        transcriptions.append(mono_transcription)

    final_transcription = "\n".join(transcriptions)
    logger.info(f"Transcription completed for {cache_batch_filename}")
    return final_transcription


def _process_channel(
    audio: Any, 
    sample_rate: int, 
    asr_processor: object, 
    cache_dir: str, 
    cache_batch_filename: str, 
    channel: str = None
) -> str:
    """
    Process a single audio channel and return its transcription.

    Args:
        audio: Audio data for the channel
        sample_rate: Sample rate of the audio
        asr_processor: ASR processor object
        cache_dir: Directory to store cache files
        cache_batch_filename: Name of the cache file
        channel: Channel name (optional, for stereo audio)

    Returns:
        str: Transcription for the channel
    """
    try:
        transcription = asr_processor.model.transcribe_batch(audio, sample_rate)[0]
        log_to_json(transcription, cache_dir, cache_batch_filename, channel)
        return transcription
    except Exception as e:
        logger.error(f"Error transcribing {'channel ' + channel if channel else 'audio'}: {str(e)}")
        return f"Error transcribing {'channel ' + channel if channel else 'audio'}: {str(e)}"


def transcribe_stream(audio, asr_processor, cache_dir, cache_streaming_filename):
    y, sr = convert_audio_sampling(audio)

    chunk_size = int(sr * 5)  # 5 second chunks
    transcription = ""

    for i in range(0, len(y), chunk_size):
        chunk = y[i: i + chunk_size]

        try:
            chunk_result = next(
                asr_processor.model.transcribe_stream([chunk], sample_rate=sr)
            )

            if isinstance(chunk_result, dict):
                chunk_transcription = chunk_result.get("transcription", "")
            else:
                chunk_transcription = chunk_result

            if chunk_transcription.strip():
                transcription += chunk_transcription + " "
                log_to_json(chunk_transcription.strip(), cache_dir, cache_streaming_filename)
                yield transcription.strip()
            else:
                log_to_json({"transcription": ""}, cache_dir, cache_streaming_filename)

        except StopIteration:
            logger.info("Transcription of chunk completed.")
            break
        except Exception as e:
            logger.error(f"Error processing chunk: {str(e)}")
            yield f"Error processing chunk: {str(e)}"

    if not transcription.strip():
        yield "No speech detected or transcription generated." 


def create_interface(args):
    asr_processor = ASRProcessor(args.language)

    with gr.Blocks() as demo:
        transcription_state = gr.State("")
        gr.Markdown("# Chinese/Taiwanese ASR Demo")

        with gr.Row():
            with gr.Column():
                model_choice = gr.Radio(
                    ["OpenAI Whisper Small", "Custom (Finetuned)", "Custom (PEFT)"],
                    label="Model Choice",
                    value="OpenAI Whisper Small",
                )
                use_peft = gr.Checkbox(label="Use PEFT (only for custom PEFT model)")
                mode = gr.Radio(
                    ["Batch", "Streaming"], label="Transcription Mode", value="Batch"
                )

            with gr.Column():
                batch_audio = gr.Audio(
                    type="numpy",
                    label="Batch Audio Input (Microphone or Upload)",
                    visible=True,
                )
                stream_audio = gr.Audio(
                    sources="microphone",
                    type="numpy",
                    label="Streaming Audio Input (Microphone only)",
                    visible=False,
                    streaming=True,
                )
                output_text = gr.Textbox(label="Transcription Output")
                summary_text = gr.Textbox(label="Summary", visible=True)
                transcribe_button = gr.Button("Transcribe", visible=True)
                clear_button = gr.Button("Clear", visible=True)
                summarize_button = gr.Button("Summarize", visible=True)

        def transcribe(audio, model_choice, use_peft):
            return transcribe_batch(
                audio, asr_processor, args.cache_dir, args.cache_batch_filename
            )

        def stream_transcribe(audio, model_choice, use_peft):
            if audio is None:
                yield "No audio input provided."
                return

            try:
                transcription_generator = transcribe_stream(
                    audio, asr_processor, args.cache_dir, args.cache_streaming_filename
                )
                
                while True:
                    try:
                        transcription = next(transcription_generator)
                        yield transcription
                    except StopIteration:
                        logger.info("Transcription completed.")
                        break
                    except Exception as e:
                        logger.error(f"Error during transcription: {str(e)}")
                        yield f"An error occurred during transcription: {str(e)}"
                        break

            except Exception as e:
                logger.error(f"Error in stream_transcribe: {str(e)}")
                yield f"An error occurred: {str(e)}"

        def summarize_batch_audio(transcript: str, state: str) -> str:
            summary = summarize_transcript(transcript)
            return summary

        def summarize_streaming_audio(args: Optional[Any] = args) -> str:
            summary = process_transcripts(args, args.cache_streaming_filename)
            return summary

        def clear_output():
            return "", "", ""

        def update_model(model_choice, use_peft):
            asr_processor.reset_model()
            asr_processor.initialize_model(model_choice, use_peft)

        def stop_streaming():
            print("Streaming has stopped")

        transcribe_button.click(
            fn=transcribe,
            inputs=[batch_audio, model_choice, use_peft],
            outputs=output_text,
        )

        stream_audio.stream(
            fn=stream_transcribe,
            inputs=[stream_audio, model_choice, use_peft],
            outputs=output_text,
        )
        
        stream_audio.stop_recording(
            stop_streaming
        ).then(
            fn=summarize_streaming_audio,
            inputs=[],
            outputs=summary_text
        )

        clear_button.click(
            fn=clear_output, 
            inputs=[], 
            outputs=[output_text, summary_text, transcription_state])

        summarize_button.click(
            fn=summarize_batch_audio,
            inputs=[output_text, transcription_state],
            outputs=[summary_text]
        )

        for input_component in [model_choice, use_peft]:
            input_component.change(
                fn=update_model, inputs=[model_choice, use_peft], outputs=None
            )

        def update_interface(mode):
            if mode == "Batch":
                return {
                    batch_audio: gr.update(visible=True),
                    stream_audio: gr.update(visible=False),
                    transcribe_button: gr.update(visible=True),
                    clear_button: gr.update(visible=True),
                    summarize_button: gr.update(visible=True),
                    summary_text: gr.update(visible=True)
                }
            else:  # Streaming
                return {
                    batch_audio: gr.update(visible=False),
                    stream_audio: gr.update(visible=True),
                    transcribe_button: gr.update(visible=False),
                    clear_button: gr.update(visible=True),
                    summarize_button: gr.update(visible=False),
                    summary_text: gr.update(visible=True)
                }

        mode.change(
            fn=update_interface,
            inputs=[mode],
            outputs=[batch_audio, stream_audio, transcribe_button, clear_button, summarize_button, summary_text],
        )

    return demo


def parse_args():
    parser = HfArgumentParser(GradioArguments)
    args = parser.parse_args_into_dataclasses()[0]
    return args


def main():
    args = parse_args()
    demo = create_interface(args)
    demo.launch(share=True)


if __name__ == "__main__":
    main()
