import argparse
import time
from time import strftime, localtime
import os

import evaluate
from tqdm import tqdm
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from datasets import load_dataset
import librosa
import numpy as np
from src.utils.logging import logger
import math

# Follow open_asr_leaderboard
# https://github.com/huggingface/open_asr_leaderboard/blob/main/speechbrain/run_eval.py


def get_whisper_model(model_name_or_path: str, device: str):
    """Load a pretrained Whisper model.

    Arguments:
    ---------
    model_name_or_path : str
        Name or path of the Whisper model. E.g., 'openai/whisper-small' or path to a custom model.
    device : str
        Device to run the model on. E.g., 'cpu' or 'cuda:0'.

    Returns:
    -------
    tuple
        Containing the loaded model and processor.
    """
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name_or_path).to(device)
    return model, processor


def transcribe_batch(model, processor, batch, device, language):
    """Transcribe a batch of audio.

    Arguments:
    ---------
    model : WhisperForConditionalGeneration
        Whisper model.
    processor : WhisperProcessor
        Whisper processor.
    batch : dict
        Batch containing audio data.
    device : str
        Device to run the model on.

    Returns:
    -------
    dict
        Dictionary containing transcription results and timing.
    """
    # Load audio inputs
    audio_inputs = batch["audio"]
    input_features = processor(
        audio_inputs, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)

    # Start timing
    start_time = time.time()
    # Generate transcriptions
    with torch.no_grad():
        generated_ids = model.generate(input_features, language=language)
    transcriptions = processor.batch_decode(generated_ids, skip_special_tokens=True)

    # End timing
    runtime = time.time() - start_time
    logger.info(f"Run Time: {runtime}")
    # Normalize transcriptions
    normalized_transcriptions = [normalize_text(trans) for trans in transcriptions]
    return {
        "transcription_time_s": [runtime / len(audio_inputs)] * len(audio_inputs),
        "predictions": normalized_transcriptions,
    }


def normalize_text(text):
    """Perform simple text normalization."""
    return text.lower().strip()


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = get_whisper_model(args.model_name_or_path, device)

    # Load dataset
    dataset = load_dataset(
        args.dataset_path, args.language, split=args.split, streaming=args.streaming, trust_remote_code=True
    )
    if args.max_eval_samples:
        dataset = dataset.select(range(min(args.max_eval_samples, len(dataset))))

    # Prepare dataset
    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["audio"] = librosa.resample(
            np.array(audio["array"]), orig_sr=audio["sampling_rate"], target_sr=16000
        )
        batch["audio_length_s"] = len(batch["audio"]) / 16000
        batch["norm_text"] = normalize_text(batch["sentence"])
        return batch

    remove_columns = [
        "client_id",
        "path",
        "up_votes",
        "down_votes",
        "age",
        "gender",
        "accent",
        "locale",
        "segment",
        "variant",
    ]
    dataset = dataset.map(prepare_dataset, remove_columns=remove_columns)

    # Evaluate
    all_results = {
        "audio_length_s": [],
        "transcription_time_s": [],
        "predictions": [],
        "references": [],
    }
    # Calculate total number of batches if possible
    if not args.streaming and hasattr(dataset, '__len__'):
        total_batches = math.ceil(len(dataset) / args.batch_size)
    else:
        total_batches = None  # Unknown size for streaming datasets

    data_iterator = dataset.iter(batch_size=args.batch_size)
    progress_bar = tqdm(data_iterator, total=total_batches, desc="Evaluating", dynamic_ncols=True)

    for batch in progress_bar:
        language = "chinese" if "zh" in args.language else args.language
        results = transcribe_batch(model, processor, batch, device, language)

        all_results["predictions"].extend(results["predictions"])
        all_results["transcription_time_s"].extend(results["transcription_time_s"])
       
        all_results["references"].extend(batch["norm_text"])
        all_results["audio_length_s"].extend(batch["audio_length_s"])

    # Calculate WER
    wer_metric = evaluate.load("wer")
    wer = wer_metric.compute(
        references=all_results["references"], predictions=all_results["predictions"]
    )
    wer = round(100 * wer, 2)

    # Calculate RTFx
    rtfx = round(
        sum(all_results["audio_length_s"]) / sum(all_results["transcription_time_s"]), 2
    )

    print(f"WER: {wer}%, RTFx: {rtfx}")

    # Save results
    output_dir = "evaluation_results"
    os.makedirs(output_dir, exist_ok=True)
    run_name = f"{strftime('%Y-%m-%d', localtime(time.time()))}"
    
    output_file = os.path.join(output_dir, f"{args.dataset_path.split('/')[-1]}_{args.split}_{run_name}_results.txt")
    with open(output_file, "w") as f:
        f.write(f"Model: {args.model_name_or_path}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"WER: {wer}%\n")
        f.write(f"RTFx: {rtfx}\n")
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Whisper model name or path",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="mozilla-foundation/common_voice_16_1",
        help="Dataset path",
    )
    parser.add_argument("--language", type=str, required=True, help="language name")
    parser.add_argument(
        "--split", type=str, default="test", help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="Device to run evaluation on (-1 for CPU, 0 or greater for GPU)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate",
    )
    parser.add_argument(
        "--streaming", default=False, help="Whether to stream the dataset"
    )
    args = parser.parse_args()

    main(args)
