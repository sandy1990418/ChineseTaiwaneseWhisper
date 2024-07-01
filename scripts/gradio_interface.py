import sys
from pathlib import Path
import numpy as np
import torch
import gradio as gr
from src.inference.flexible_inference import ChineseTaiwaneseASRInference
import logging
import soundfile as sf
import os
import tempfile

project_root = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def transcribe_audio(audio, model_choice, use_peft, inference_type):
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        if model_choice == "Custom (Finetuned)":
            model_path = "./whisper-finetuned-zh-tw"
        elif model_choice == "Custom (PEFT)":
            model_path = "./whisper-peft-finetuned-zh-tw"
            use_peft = True
        else:
            model_path = "openai/whisper-small"
        
        logger.info(f"Loading model from: {model_path}")
        inference = ChineseTaiwaneseASRInference(model_path, device=device, use_peft=use_peft, language="chinese")
        
        if audio is None:
            return "No audio input provided."

        logger.info(f"Audio input type: {type(audio)}")
        
        if isinstance(audio, str):
            logger.info(f"Loading audio from file: {audio}")
            audio_data, samplerate = sf.read(audio)
        elif isinstance(audio, (tuple, list)):
            logger.info("Processing microphone input")
            samplerate, audio_data = audio
        else:
            logger.info("Processing direct audio data")
            audio_data = audio

        # Ensure audio is in float32 format and normalize
        audio_data = audio_data.astype(np.float32)
        audio_data = audio_data / np.max(np.abs(audio_data))

        logger.info(f"Audio data shape: {audio_data.shape}, Sample rate: {samplerate}")
        logger.info(f"Audio data min: {np.min(audio_data)}, max: {np.max(audio_data)}")

        if inference_type == "Batch":
            transcription = inference.transcribe_batch([audio_data])[0]
        else:  # Streaming
            transcription = " ".join(list(inference.transcribe_stream(audio_data)))
        
        logger.info(f"Transcription result: {transcription}")
        return transcription
    except FileNotFoundError:
        return "Error: Custom model not found. Please ensure the model file exists."
    except Exception as e:
        logger.error(f"Error in transcribe_audio: {e}", exc_info=True)
        return f"An error occurred: {str(e)}"

def process_audio(audio):
    if audio is None:
        return None
    if isinstance(audio, str):  # It's a file path
        return audio
    elif isinstance(audio, (tuple, list)):  # It's from microphone
        # Save the audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            sf.write(temp_audio.name, audio[1], audio[0])
        return temp_audio.name
    else:
        raise ValueError(f"Unexpected audio type: {type(audio)}")

def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Chinese/Taiwanese Whisper ASR Demo")
        gr.Markdown("Upload an audio file or use microphone to record audio for transcription.")
        
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(label="Audio Input")
            
            with gr.Column():
                model_choice = gr.Radio(["OpenAI Whisper Small", "Custom (Finetuned)", "Custom (PEFT)"], 
                                        label="Model Choice", value="OpenAI Whisper Small")
                use_peft = gr.Checkbox(label="Use PEFT (only for custom PEFT model)")
                inference_type = gr.Radio(["Batch", "Streaming"], label="Inference Type", value="Batch")
                transcribe_button = gr.Button("Transcribe")
        
        output_text = gr.Textbox(label="Transcription Output")
        
        transcribe_button.click(
            fn=transcribe_audio,
            inputs=[audio_input, model_choice, use_peft, inference_type],
            outputs=output_text
        )

    demo.launch(share=True)

if __name__ == "__main__":
    create_gradio_interface()