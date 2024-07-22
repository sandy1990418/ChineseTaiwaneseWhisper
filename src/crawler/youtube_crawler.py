import os
import re
import json
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from tqdm import tqdm
import soundfile as sf
from pathlib import Path
# import pandas as pd
import subprocess
import sys
import logging
from datasets import Dataset, IterableDataset
from src.config import CrawlerArgs
import librosa
from transformers import HfArgumentParser
import numpy as np
from collections import defaultdict
import gc


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_ffmpeg(ffmpeg_path=None):
    """Check if FFmpeg is installed and accessible."""
    try:
        if ffmpeg_path:
            subprocess.run(
                [ffmpeg_path, "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.run(
                ["ffmpeg", "-version"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return True
    except FileNotFoundError:
        return False


def extract_playlist_id(url):
    """Extract playlist ID from a YouTube playlist URL."""
    playlist_id_match = re.search(r"(?:list=)([a-zA-Z0-9_-]+)", url)
    return playlist_id_match.group(1) if playlist_id_match else None


def download_youtube_audio_and_subtitles(
    video_id, output_dir, ffmpeg_path=None, file_prefix="", play_idx=0, file_index=0
):
    """Download YouTube audio and subtitles for a given video ID."""
    audio_dir = os.path.join(output_dir, "audio")
    subtitle_dir = os.path.join(output_dir, "subtitles")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(subtitle_dir, exist_ok=True)

    file_name = f"{file_prefix}_{play_idx:04d}_{file_index:04d}"
    audio_file = os.path.join(audio_dir, f"{file_name}")
    subtitle_file = os.path.join(subtitle_dir, f"{file_name}.json")

    # Download audio if it doesn't exist
    if not os.path.exists(audio_file):
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "wav",
                    "preferredquality": "192",
                }
            ],
            "outtmpl": audio_file,
        }
        if ffmpeg_path:
            ydl_opts["ffmpeg_location"] = ffmpeg_path
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        except yt_dlp.utils.DownloadError as e:
            logger.error(f"Error downloading audio for video {video_id}: {e}")
            return None

    # Get subtitles if they don't exist
    if not os.path.exists(subtitle_file):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(
                video_id, languages=["zh-TW", "zh-CN", "zh"]
            )
            with open(subtitle_file, "w", encoding="utf-8") as f:
                json.dump(transcript, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error getting transcript for video {video_id}: {e}")
            return None

    return audio_file, subtitle_file


def crawl_youtube_playlist(
    playlist_url, play_idx, output_dir, ffmpeg_path=None, file_prefix="", batch_size=20
):
    """Crawl videos in a YouTube playlist in batches."""
    playlist_id = extract_playlist_id(playlist_url)
    if not playlist_id:
        logger.error(f"Invalid playlist URL: {playlist_url}")
        return [], []

    ydl_opts = {
        "extract_flat": True,
        "quiet": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        playlist_dict = ydl.extract_info(
            f"https://www.youtube.com/playlist?list={playlist_id}", download=False
        )

    all_audio_files = []
    all_subtitle_files = []
    batch_audio_files = []
    batch_subtitle_files = []

    for index, video in enumerate(playlist_dict["entries"]):
        video_id = video["id"]
        logger.info(f"Processing video: {video_id}")
        result = download_youtube_audio_and_subtitles(
            video_id, output_dir, ffmpeg_path, file_prefix, play_idx, index
        )
        if result:
            audio_file, subtitle_file = result
            batch_audio_files.append(audio_file)
            batch_subtitle_files.append(subtitle_file)

        if (
            len(batch_audio_files) == batch_size
            or index == len(playlist_dict["entries"]) - 1
        ):
            all_audio_files.extend(batch_audio_files)
            all_subtitle_files.extend(batch_subtitle_files)
            yield batch_audio_files, batch_subtitle_files
            batch_audio_files = []
            batch_subtitle_files = []


# def create_dataset(audio_files, subtitle_files):
#     """Create a Hugging Face dataset from a batch of files."""
#     data = []
#     for audio_file, subtitle_file in zip(audio_files, subtitle_files):
#         with open(subtitle_file, "r", encoding="utf-8") as f:
#             subtitle = json.load(f)
#         text = " ".join([entry["text"] for entry in subtitle])
#         data.append({"audio": f"{audio_file}.wav", "sentence": text})

#     df = pd.DataFrame(data)
#     return Dataset.from_pandas(df)

# def slice_audio(audio: np.ndarray, start: float, duration: float, sampling_rate: int) -> np.ndarray:
#     """Slice audio array based on start time and duration."""
#     start_sample = int(start * sampling_rate)
#     end_sample = int((start + duration) * sampling_rate)
#     return audio[start_sample:end_sample]


def create_dataset(audio_files, subtitle_files):
    """Create a Hugging Face dataset from a batch of files."""

    def generate_dataset_format():
        for audio_file, subtitle_file in zip(audio_files, subtitle_files):
            # Load audio file
            audio_array, sampling_rate = librosa.load(
                f"{audio_file}.wav", sr=None, mono=True, dtype=np.float32
            )
            # audio_array, sampling_rate = sf.read(f"{audio_file}.wav")

            # Read subtitle file
            with open(subtitle_file, "r", encoding="utf-8") as f:
                subtitle = json.load(f)

            # Process each subtitle entry
            for entry_idx, entry in enumerate(subtitle):
                file_name = os.path.basename(audio_file)
                # sliced_audio = slice_audio(audio_array, entry["start"], entry["duration"], sampling_rate)
                yield {
                    "client_id": f"{file_name}_{entry_idx}",
                    "path": f"{audio_file}.wav",
                    # "audio": {
                    #     "path": f"{audio_file}.wav",
                    #     "array": sliced_audio,
                    #     "sampling_rate": sampling_rate
                    # },
                    "sentence": entry["text"],
                    "start": entry["start"],
                    "end": entry["start"] + entry["duration"],
                    "duration": entry["duration"],
                }
            del audio_array
            gc.collect()
        # Create the dataset

    return IterableDataset.from_generator(generate_dataset_format)


def iterable_to_dataset(iterable_dataset):
    """Convert an IterableDataset to a regular Dataset."""
    data_dict = defaultdict(list)
    for item in iterable_dataset:
        for key, value in item.items():
            data_dict[key].append(value)
    return Dataset.from_dict(data_dict)


def append_to_json(segments, json_path):
    """
    Append segments to a single JSON file.
    If the file doesn't exist, it creates a new one.
    """
    if os.path.exists(json_path):
        with open(json_path, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data.extend(segments)
            f.seek(0)
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.truncate()
    else:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=4)


def process_segment(segment, audio, sr, audio_file, split_audio_dir, segments):
    start_time = float(segment[0]['start'])
    end_time = float(segment[-1]['start']) + float(segment[-1]['duration'])
    duration = end_time-start_time
    
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)
    split_audio = audio[start_sample:end_sample]
    
    segment_filename = f"{Path(audio_file).stem}_{len(segments):04d}.wav"
    segment_path = os.path.join(split_audio_dir, segment_filename)
    sf.write(segment_path, split_audio, sr)

    timestamp = []
    for sub in segment:
        sub_time = {
            "start": float(sub['start']),
            "end":  float(sub['start']+sub['duration']),
            "text": sub['text']
        }
        timestamp.append(sub_time)

    segments.append({
        "audio_path": segment_path,
        "start": start_time,
        "end": end_time,
        "duration": duration,
        "text": " ".join([s['text'].strip() for s in segment]),
        "timestamp": timestamp
    })


def process_audio_file(audio_file, subtitle_file, output_dir, json_path, max_duration=30):
    """
    Process a single audio file: split it and append segment information to the JSON file.
    Only process if the subtitle file exists and skip empty segments.
    """
    # Check if subtitle file exists
    if not subtitle_file or not os.path.exists(subtitle_file):
        logger.warning(f"Subtitle file not found for {audio_file}. Skipping this audio file.")
        return

    # Load the audio file
    audio, sr = librosa.load(audio_file, sr=None, mono=True)
    
    # Load subtitle data
    with open(subtitle_file, 'r', encoding='utf-8') as f:
        subtitles = json.load(f)
    
    # Create output directories
    split_audio_dir = os.path.join(output_dir, "split_audio")
    os.makedirs(split_audio_dir, exist_ok=True)
    
    segments = []
    current_segment = []
    current_start = 0
    # current_end = 0
        
    for subtitle in subtitles:
        # Skip empty subtitles
        if not subtitle['text'].strip():
            continue

        start_time = float(subtitle['start'])
        duration = float(subtitle['duration'])
        end_time = start_time + duration
        if end_time - current_start >= max_duration:
            if current_segment:
                process_segment(current_segment[:-1], audio, sr, audio_file, split_audio_dir, segments)
            current_segment = [subtitle]
            current_start = start_time
        else:
            current_segment.append(subtitle)
            # current_end = end_time

    # Process the last segment if it exists
    if current_segment:
        process_segment(current_segment, audio, sr, audio_file, split_audio_dir, segments)

    if segments:
        append_to_json(segments, json_path)
    else:
        logger.warning(f"No valid segments found for {audio_file}. Skipping this audio file.")
        

def create_dataset_from_json(json_file):
    """Create a Hugging Face dataset from a JSON file."""
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    return Dataset.from_dict(
        {
            "client_id": [
                f"{os.path.basename(item['audio_path'])}_{i}"
                for i, item in enumerate(data)
            ],
            "path": [item["audio_path"] for item in data],
            "sentence": [item["text"] for item in data],
            "start": [item["start"] for item in data],
            "end": [item["end"] for item in data],
            "duration": [item["end"] - item["start"] for item in data],
        }
    )


def convert_dataset_to_json(dataset, output_file):
    """Convert a Hugging Face dataset to a JSON file that can be read by load_dataset."""
    data = []
    for item in tqdm(dataset, desc="Converting dataset to JSON"):
        data.append(
            {
                "client_id": item["client_id"],
                "path": item["path"],
                "sentence": item["sentence"],
                "start": item["start"],
                "end": item["end"],
                "duration": item["duration"],
            }
        )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"Dataset saved as JSON: {output_file}")


def parse_args():
    parser = HfArgumentParser(CrawlerArgs)
    args = parser.parse_args_into_dataclasses()[0]
    return args


def main(args):
    if not args.output_dir:
        logger.error("Output directory must be specified.")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    if not check_ffmpeg(args.ffmpeg_path):
        logger.error("Error: FFmpeg is not installed or not in the system PATH.")
        logger.error(
            "Please install FFmpeg and make sure it's accessible from the command line."
        )
        logger.error("You can download FFmpeg from: https://ffmpeg.org/download.html")
        logger.error(
            "After installation, you may need to restart your terminal or computer."
        )
        logger.error(
            "Alternatively, you can specify the path to FFmpeg using the --ffmpeg_path argument."
        )
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    json_path = os.path.join(args.output_dir, f"{args.dataset_name}.json")

    if args.playlist_urls:
        # Process YouTube playlists
        for play_idx, playlist_url in enumerate(args.playlist_urls):
            logger.info(f"Processing playlist: {playlist_url}")
            for batch_num, (batch_audio_files, batch_subtitle_files) in enumerate(
                crawl_youtube_playlist(
                    playlist_url,
                    play_idx,
                    args.output_dir,
                    args.ffmpeg_path,
                    args.file_prefix,
                    args.batch_size,
                )
            ):
                logger.info(f"Processing batch {batch_num + 1}")

                # Process and split audio files
                for audio_file, subtitle_file in zip(
                    batch_audio_files, batch_subtitle_files
                ):
                    process_audio_file(
                        f"{audio_file}.wav", subtitle_file, args.output_dir, json_path
                    )

    if args.audio_dir:
        # Process existing audio files
        audio_files = [
            os.path.join(args.audio_dir, f)
            for f in os.listdir(args.audio_dir)
            if f.endswith(".wav")
        ]
        subtitle_files = [
            os.path.join(os.path.dirname(args.audio_dir), "subtitles", f.replace(".wav", ".json"))
            for f in os.listdir(args.audio_dir)
            if f.endswith(".wav")
        ]
        for audio_file, subtitle_file in tqdm(
            zip(audio_files, subtitle_files),
            total=len(audio_files),
            desc="Processing audio files",
        ):
            process_audio_file(audio_file, subtitle_file, args.output_dir, json_path)

    logger.info(f"All segments saved to: {json_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
