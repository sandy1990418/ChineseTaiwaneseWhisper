import os
import re
import json
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
# import pandas as pd
import subprocess
import sys
import logging
from datasets import Dataset, concatenate_datasets, IterableDataset
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
            audio_array, sampling_rate = librosa.load(f"{audio_file}.wav", 
                                                      sr=None, 
                                                      mono=True,  
                                                      dtype=np.float32)
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
                    "end": entry["start"]+entry["duration"],
                    "duration": entry["duration"]
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


def parse_args():
    parser = HfArgumentParser(CrawlerArgs)
    args = parser.parse_args_into_dataclasses()[0]
    return args


def main(args):
    if not args.playlist_urls:
        logger.error("At least one playlist URL must be provided.")
        sys.exit(1)

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

    all_datasets = []

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
            logger.info(f"Creating dataset for batch {batch_num + 1}")
            batch_iterable_dataset = create_dataset(batch_audio_files, batch_subtitle_files)
            batch_dataset = iterable_to_dataset(batch_iterable_dataset)
            all_datasets.append(batch_dataset)

            # Save intermediate dataset
            intermediate_dataset = concatenate_datasets(all_datasets)
            intermediate_dataset.save_to_disk(
                os.path.join(args.output_dir, f"{args.dataset_name}")
            )
            logger.info(f"Intermediate dataset saved: {args.dataset_name}")

        # Combine all datasets and save the final result
        final_dataset = concatenate_datasets(all_datasets)
        final_dataset.save_to_disk(os.path.join(args.output_dir, args.dataset_name))
        logger.info(
            f"Final dataset saved to {os.path.join(args.output_dir, args.dataset_name)}"
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
