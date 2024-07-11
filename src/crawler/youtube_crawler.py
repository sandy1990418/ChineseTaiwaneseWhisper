import os
import re
import json
import random
import time
from typing import List, Tuple
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
import pandas as pd
import argparse
import subprocess
import sys
import logging
from datasets import Dataset, concatenate_datasets
from src.config.train_config import CrawlerArgs
import requests
from fake_useragent import UserAgent
from tenacity import retry, stop_after_attempt, wait_exponential

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


def get_random_proxy() -> str:
    """Get a random proxy from a list of free proxies."""
    response = requests.get("https://free-proxy-list.net/")
    proxies = re.findall(r"<td>\d+\.\d+\.\d+\.\d+</td><td>\d+</td>", response.text)
    return random.choice(proxies).split("<td>")[1].split("</td>")[0]


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def download_youtube_audio_and_subtitles(
    video_id: str,
    output_dir: str,
    ffmpeg_path: str = None,
    file_prefix: str = "",
    file_index: int = 0,
) -> Tuple[str, str]:
    """Download YouTube audio and subtitles for a given video ID with retry logic."""
    audio_dir = os.path.join(output_dir, "audio")
    subtitle_dir = os.path.join(output_dir, "subtitles")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(subtitle_dir, exist_ok=True)

    file_name = f"{file_prefix}_{file_index:04d}"
    audio_file = os.path.join(audio_dir, f"{file_name}")
    subtitle_file = os.path.join(subtitle_dir, f"{file_name}.json")

    ua = UserAgent()
    proxy = get_random_proxy()

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
            "proxy": f"http://{proxy}",
            "user_agent": ua.random,
            "sleep_interval": random.randint(1, 3),
            "max_sleep_interval": 5,
        }
        if ffmpeg_path:
            ydl_opts["ffmpeg_location"] = ffmpeg_path
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        except yt_dlp.utils.DownloadError as e:
            logger.error(f"Error downloading audio for video {video_id}: {e}")
            raise

    if not os.path.exists(subtitle_file):
        try:
            transcript = YouTubeTranscriptApi.get_transcript(
                video_id, languages=["zh-TW", "zh-CN", "zh"], proxies={"http": proxy}
            )
            with open(subtitle_file, "w", encoding="utf-8") as f:
                json.dump(transcript, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error getting transcript for video {video_id}: {e}")
            raise

    return audio_file, subtitle_file


def crawl_youtube_playlist(
    playlist_url: str,
    output_dir: str,
    ffmpeg_path: str = None,
    file_prefix: str = "",
    batch_size: int = 20,
) -> List[Tuple[List[str], List[str]]]:
    """Crawl videos in a YouTube playlist in batches with improved robustness."""
    playlist_id = extract_playlist_id(playlist_url)
    if not playlist_id:
        logger.error(f"Invalid playlist URL: {playlist_url}")
        return []

    ua = UserAgent()
    proxy = get_random_proxy()

    ydl_opts = {
        "extract_flat": True,
        "quiet": True,
        "proxy": f"http://{proxy}",
        "user_agent": ua.random,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        playlist_dict = ydl.extract_info(
            f"https://www.youtube.com/playlist?list={playlist_id}", download=False
        )

    batches = []
    batch_audio_files = []
    batch_subtitle_files = []

    for index, video in enumerate(playlist_dict["entries"]):
        video_id = video["id"]
        logger.info(f"Processing video: {video_id}")

        try:
            audio_file, subtitle_file = download_youtube_audio_and_subtitles(
                video_id, output_dir, ffmpeg_path, file_prefix, index
            )
            batch_audio_files.append(audio_file)
            batch_subtitle_files.append(subtitle_file)
        except Exception as e:
            logger.error(f"Failed to process video {video_id}: {e}")
            continue

        if (
            len(batch_audio_files) == batch_size
            or index == len(playlist_dict["entries"]) - 1
        ):
            batches.append((batch_audio_files, batch_subtitle_files))
            batch_audio_files = []
            batch_subtitle_files = []

        # Add a random delay between video downloads
        time.sleep(random.uniform(1, 3))

    return batches


def create_dataset(audio_files, subtitle_files, output_file):
    """Create and save a Hugging Face dataset."""
    data = []
    for audio_file, subtitle_file in zip(audio_files, subtitle_files):
        with open(subtitle_file, "r", encoding="utf-8") as f:
            subtitle = json.load(f)
        text = " ".join([entry["text"] for entry in subtitle])
        data.append({"audio": f"{audio_file}.wav", "sentence": text})

    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)

    # Save the dataset
    dataset.save_to_disk(output_file)

    logger.info(f"Dataset saved to {output_file}")


def parse_args() -> CrawlerArgs:
    parser = argparse.ArgumentParser(
        description="YouTube Audio Crawler and Dataset Creator"
    )
    parser.add_argument(
        "--playlist_urls",
        nargs="+",
        required=True,
        help="YouTube playlist URLs to crawl",
    )
    parser.add_argument(
        "--output_dir",
        default="./output",
        help="Directory to save audio files and dataset",
    )
    parser.add_argument(
        "--dataset_name",
        default="youtube_dataset",
        help="Name of the output dataset file",
    )
    parser.add_argument("--ffmpeg_path", help="Path to FFmpeg executable")
    parser.add_argument(
        "--file_prefix", default="youtube", help="Prefix for audio and subtitle files"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="Number of videos to process before saving",
    )

    args = parser.parse_args()
    return CrawlerArgs(**vars(args))


def main(args: CrawlerArgs):
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

    for playlist_url in args.playlist_urls:
        logger.info(f"Processing playlist: {playlist_url}")
        batches = crawl_youtube_playlist(
            playlist_url,
            args.output_dir,
            args.ffmpeg_path,
            args.file_prefix,
            args.batch_size,
        )

        for batch_num, (batch_audio_files, batch_subtitle_files) in enumerate(batches):
            logger.info(f"Creating dataset for batch {batch_num + 1}")
            batch_dataset = create_dataset(batch_audio_files, batch_subtitle_files)
            all_datasets.append(batch_dataset)

            # Save intermediate dataset
            intermediate_dataset = concatenate_datasets(all_datasets)
            intermediate_dataset.save_to_disk(
                os.path.join(
                    args.output_dir, f"{args.dataset_name}"
                )
            )
            logger.info(
                f"Intermediate dataset saved: {args.dataset_name}"
            )

        # Add a longer delay between processing playlists
        time.sleep(random.uniform(5, 10))

    # Combine all datasets and save the final result
    final_dataset = concatenate_datasets(all_datasets)
    final_dataset.save_to_disk(os.path.join(args.output_dir, args.dataset_name))
    logger.info(
        f"Final dataset saved to {os.path.join(args.output_dir, args.dataset_name)}"
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
