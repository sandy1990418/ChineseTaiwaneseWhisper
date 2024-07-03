import gradio as gr
import numpy as np
import torch
from src.inference.flexible_inference import ChineseTaiwaneseASRInference
from scipy import signal
import os 
from datetime import datetime

cache_dir = os.path.join(os.getcwd(), "asr_transcription_streaming_cache")
os.makedirs(cache_dir, exist_ok=True)


class ASRProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.initialize_model("OpenAI Whisper Small", False)

    def initialize_model(self, model_choice, use_peft):
        if model_choice == "Custom (Finetuned)":
            model_path = "./whisper-finetuned-zh-tw"
        elif model_choice == "Custom (PEFT)":
            model_path = "./whisper-peft-finetuned-zh-tw"
        else:
            model_path = "openai/whisper-small"

        self.model = ChineseTaiwaneseASRInference(
            model_path, 
            device=self.device, 
            use_peft=use_peft, 
            language="chinese"
        )


asr_processor = ASRProcessor()


def log_to_file(message):
    log_file = os.path.join(cache_dir, "asr_log.txt")
    with open(log_file, "a") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")


def convert_audio_sampling(audio):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    y = resample_audio(y, sr, 16000)
    sr = 16000

    return y, sr


def resample_audio(y, orig_sr, target_sr):
    if orig_sr != target_sr:
        num_samples = int(len(y) * float(target_sr) / orig_sr)
        y = signal.resample(y, num_samples)
    return y


def transcribe_batch(audio, model_choice, use_peft):
    if audio is None:
        return "No audio input provided."
    
    y, _ = convert_audio_sampling(audio)

    transcription = asr_processor.model.transcribe_batch([y])[0]
    return transcription


def transcribe_stream(audio, model_choice, use_peft):
    if audio is None:
        return "No audio input provided."
    
    y, sr = convert_audio_sampling(audio)

    chunk_size = int(sr * 5)  # 5 second chunks
    transcription = ""
    
    for i in range(0, len(y), chunk_size):
        chunk = y[i:i+chunk_size]
        chunk_transcription = next(asr_processor.model.transcribe_stream([chunk], sample_rate=sr))
        transcription += chunk_transcription + " "

        log_to_file(chunk_transcription)
        yield transcription.strip()


def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Chinese/Taiwanese ASR Demo")
        
        with gr.Row():
            with gr.Column():
                model_choice = gr.Radio(["OpenAI Whisper Small", "Custom (Finetuned)", "Custom (PEFT)"], 
                                        label="Model Choice", value="OpenAI Whisper Small")
                use_peft = gr.Checkbox(label="Use PEFT (only for custom PEFT model)")
                mode = gr.Radio(["Batch", "Streaming"], label="Transcription Mode", value="Batch")
            
            with gr.Column():
                batch_audio = gr.Audio(type="numpy", label="Batch Audio Input (Microphone or Upload)", visible=True)
                stream_audio = gr.Audio(sources="microphone", type="numpy", label="Streaming Audio \
                                        Input (Microphone only)", visible=False, streaming=True)
                output_text = gr.Textbox(label="Transcription Output")
                transcribe_button = gr.Button("Transcribe", visible=True)
                clear_button = gr.Button("Clear", visible=False)

        def transcribe(audio, model_choice, use_peft):
            return transcribe_batch(audio, model_choice, use_peft)

        def stream_transcribe(audio, model_choice, use_peft):
            for transcription in transcribe_stream(audio, model_choice, use_peft):
                yield transcription

        def clear_output():
            return ""

        transcribe_button.click(
            fn=transcribe,
            inputs=[batch_audio, model_choice, use_peft],
            outputs=output_text
        )

        stream_audio.stream(
            fn=stream_transcribe,
            inputs=[stream_audio, model_choice, use_peft],
            outputs=output_text
        )

        clear_button.click(
            fn=clear_output,
            inputs=[],
            outputs=output_text
        )

        def update_interface(mode):
            if mode == "Batch":
                return {
                    batch_audio: gr.update(visible=True),
                    stream_audio: gr.update(visible=False),
                    transcribe_button: gr.update(visible=True),
                    clear_button: gr.update(visible=False)
                }
            else:  # Streaming
                return {
                    batch_audio: gr.update(visible=False),
                    stream_audio: gr.update(visible=True),
                    transcribe_button: gr.update(visible=False),
                    clear_button: gr.update(visible=True)
                }

        mode.change(
            fn=update_interface,
            inputs=[mode],
            outputs=[batch_audio, stream_audio, transcribe_button, clear_button]
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)