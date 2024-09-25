import os
import re
import json
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from tqdm import tqdm
import soundfile as sf
from pathlib import Path

import subprocess
import logging
from datasets import Dataset, IterableDataset
from src.config import CrawlerArgs
import librosa
from transformers import HfArgumentParser
import numpy as np
from collections import defaultdict
import gc
import requests
from typing import Optional, Tuple, Dict, Any
from dotenv import load_dotenv
from pydub import AudioSegment
import openai

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB in bytes
INITIAL_CHUNK_DURATION = 10 * 60 * 1000  # 10 minutes in milliseconds

# TODO:
# 1. single audio file
# 2. single url
# 3. strcture adjust
# 4. test example


class YoutubeCrawler:
    def __init__(self, args: CrawlerArgs):
        self.args = args
        self.output_dir = args.output_dir
        self.ffmpeg_path = args.ffmpeg_path
        self.file_prefix = args.file_prefix
        self.batch_size = args.batch_size
        self.max_duration = args.max_duration

    def crawl(self):
        self._check_ffmpeg()
        self._create_output_dirs()
        json_path = os.path.join(
            self.output_dir, self.file_prefix, f"{self.args.dataset_name}.json"
        )
        if self.args.playlist_urls:
            logger.info(f"Processing YouTube playlists: {self.args.playlist_urls}")
            for play_idx, playlist_url in enumerate(self.args.playlist_urls):
                self._process_youtube_playlist(play_idx, playlist_url, json_path)

        if self.args.audio_dir:
            logger.info(f"Processing existing audio files from: {self.args.audio_dir}")
            self._process_audio_file(self.audio_dir, self.subtitle_dir, json_path)

        logger.info(f"All segments saved to: {json_path}")

    def _check_ffmpeg(self, ffmpeg_path=None):
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

    def _create_output_dirs(self):
        self.audio_dir = os.path.join(self.output_dir, self.file_prefix, "audio")
        self.subtitle_dir = os.path.join(self.output_dir, self.file_prefix, "subtitles")
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.subtitle_dir, exist_ok=True)

    def _extract_playlist_id(self, url: str) -> Optional[str]:
        playlist_id_match = re.search(r"(?:list=)([a-zA-Z0-9_-]+)", url)
        return playlist_id_match.group(1) if playlist_id_match else None

    def _process_youtube_playlist(self, play_idx, playlist_url, json_path):
        """Crawl videos in a YouTube playlist in batches."""

        playlist_id = self._extract_playlist_id(playlist_url)

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

        for index, video in enumerate(playlist_dict["entries"]):
            video_id = video["id"]
            logger.info(f"Processing video: {video_id}")
            result = self._download_youtube_audio_and_subtitles(
                video_id, play_idx, index
            )
            if result:
                audio_file, subtitle_file = result
                self._process_audio_file(audio_file, subtitle_file, json_path)

            if index + 1 >= self.batch_size:
                logger.info(f"Reached batch size limit of {self.batch_size}. Stopping.")
                break

    def _download_youtube_audio_and_subtitles(
        self, video_id: str, play_idx: int, file_index: int
    ) -> Optional[Tuple[Path, Path]]:
        file_name = f"{self.file_prefix}_{play_idx:04d}_{file_index:04d}"
        audio_file = Path(os.path.join(self.audio_dir, f"{file_name}.mp3"))
        subtitle_file = Path(os.path.join(self.subtitle_dir, f"{file_name}.json"))

        if not audio_file.exists():
            try:
                audio_file = self._download_audio_with_yt_dlp(video_id, audio_file)
            except yt_dlp.utils.DownloadError:
                try:
                    audio_file = self._download_audio_with_cobalt(video_id, audio_file)
                except Exception as e:
                    logger.error(
                        f"Error downloading audio for video {video_id} using Cobalt API: {e}"
                    )
                    return None
        # Convert mp3 to wav if needed
        wav_file = audio_file.with_suffix(".wav")
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    str(audio_file),
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    str(wav_file),
                ],
                check=True,
            )
            audio_file = wav_file
            os.remove(
                str(audio_file.with_suffix(".mp3"))
            )  # Remove the original mp3 file
        except subprocess.CalledProcessError as e:
            logger.error(f"Error converting mp3 to wav: {e}")
            return None

        if not subtitle_file.exists():
            try:
                transcript = YouTubeTranscriptApi.get_transcript(
                    video_id, languages=["zh-TW", "zh-CN", "zh"]
                )
                with open(subtitle_file, "w", encoding="utf-8") as f:
                    json.dump(transcript, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Error getting transcript for video {video_id}: {e}")
                logger.info(
                    f"Attempting to transcribe audio using OpenAI API for video {video_id}"
                )
                video_info = {
                    "video_id": video_id,
                    "video_title": f"video_{video_id}",
                    "channel_title": "unknown_channel",
                }
                transcription_result = self.process_and_transcribe_audio(
                    video_info, str(audio_file)
                )
                if "error" not in transcription_result:
                    with open(subtitle_file, "w", encoding="utf-8") as f:
                        json.dump(
                            transcription_result["transcript"],
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )
                else:
                    logger.error(
                        f"Failed to transcribe audio for video {video_id}: {transcription_result['error']}"
                    )
                    return None
        return audio_file, subtitle_file

    def _download_audio_with_yt_dlp(self, video_id: str, audio_file: Path) -> Path:
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
        if self.ffmpeg_path:
            ydl_opts["ffmpeg_location"] = self.ffmpeg_path
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
        logger.info(f"Download successful! {audio_file}")
        return audio_file

    def _download_audio_with_cobalt(self, video_id: str, audio_file: Path) -> Path:
        # Thanks to Cobalt! Your work is truly great.
        # https://github.com/imputnet/cobalt
        logger.info("Initiating download using Cobalt API.")

        url = "https://olly.imput.net/api/json"
        params = {
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "isAudioOnly": True,
        }
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        # Make the API request
        response = requests.post(url, json=params, headers=headers)

        if response.status_code == 200:
            result = response.json()
            download_url = result["url"]
            # Step 2: Download the audio content from the stream
            logger.info("Start to stream download using Cobalt API.")
            with requests.get(download_url, stream=True) as stream_response:
                stream_response.raise_for_status()
                os.makedirs(self.audio_dir, exist_ok=True)
                with open(audio_file, "wb") as file:
                    for chunk in stream_response.iter_content(chunk_size=8192):
                        file.write(chunk)

            logger.info("Download successful!")
            return audio_file

    def _process_audio_file(self, audio_file: str, subtitle_file: str, json_path: str):
        if not Path(subtitle_file).exists():
            logger.warning(
                f"Subtitle file not found for {audio_file}. Skipping this audio file."
            )
            return

        audio, sr = librosa.load(audio_file, sr=None, mono=True)

        split_audio_dir = os.path.join(self.output_dir, self.file_prefix, "split_audio")
        os.makedirs(split_audio_dir, exist_ok=True)

        with open(subtitle_file, "r", encoding="utf-8") as f:
            subtitles = json.load(f)

        segments = []
        current_segment = []
        current_start = 0

        for subtitle in subtitles:
            if not subtitle["text"].strip():
                continue

            start_time = float(subtitle["start"])
            duration = float(subtitle["duration"])
            end_time = start_time + duration

            if end_time - current_start >= self.max_duration:
                if current_segment:
                    self._process_segment(
                        current_segment[:-1],
                        audio,
                        sr,
                        audio_file,
                        split_audio_dir,
                        segments,
                    )
                current_segment = [subtitle]
                current_start = start_time
            else:
                current_segment.append(subtitle)

        if current_segment:
            self._process_segment(
                current_segment, audio, sr, audio_file, split_audio_dir, segments
            )
        if segments:
            self._append_to_json(segments, json_path)
        else:
            logger.warning(
                f"No valid segments found for {audio_file}. Skipping this audio file."
            )

    def _process_segment(
        self, segment, audio, sr, audio_file, split_audio_dir, segments
    ):
        if not segment:
            logger.warning(
                f"Empty segment encountered for {audio_file}. Skipping this segment."
            )
            return

        start_time = float(segment[0]["start"])
        end_time = float(segment[-1]["start"]) + float(segment[-1]["duration"])
        duration = end_time - start_time

        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        split_audio = audio[start_sample:end_sample]

        segment_filename = f"{audio_file.stem}_{len(segments):04d}.wav"
        segment_path = os.path.join(split_audio_dir, segment_filename)
        # os.makedirs(os.path.dirname(segment_path), exist_ok=True)
        sf.write(segment_path, split_audio, sr)

        timestamp = [
            {
                "start": float(sub["start"]),
                "end": float(sub["start"] + sub["duration"]),
                "text": sub["text"],
            }
            for sub in segment
        ]

        segments.append(
            {
                "audio_path": str(segment_path),
                "start": start_time,
                "end": end_time,
                "duration": duration,
                "text": " ".join([s["text"].strip() for s in segment]),
                "timestamp": timestamp,
            }
        )

    def process_and_transcribe_audio(
        self, video_info: Dict[str, str], audio_file: str
    ) -> Dict[str, Any]:
        self.client = openai.OpenAI()
        try:
            audio = AudioSegment.from_wav(audio_file)
            transcripts = []
            start = 0

            while start < len(audio):
                end = min(start + INITIAL_CHUNK_DURATION, len(audio))

                while True:
                    chunk = audio[start:end]
                    chunk_file = os.path.join(self.output_dir, "temp_chunk.mp3")
                    chunk.export(chunk_file, format="mp3", bitrate="64k")

                    if os.path.getsize(chunk_file) <= MAX_FILE_SIZE:
                        break

                    end = start + (end - start) // 2
                    os.remove(chunk_file)

                    if end - start < 1000:  # Minimum 1 second chunk
                        raise ValueError("Unable to create a small enough chunk")

                with open(chunk_file, "rb") as audio_chunk:
                    response = self.client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_chunk,
                        response_format="verbose_json",
                    )

                transcripts.extend(response.segments)
                os.remove(chunk_file)

                start = end

            result = {
                "video_id": video_info["video_id"],
                "video_title": video_info["video_title"],
                "channel_title": video_info["channel_title"],
                "transcript": [
                    {
                        "text": segment["text"],
                        "start": segment["start"],
                        "duration": segment["end"] - segment["start"],
                    }
                    for segment in transcripts
                ],
            }
            return result

        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return {"video_id": video_info["video_id"], "error": str(e)}

    @staticmethod
    def _append_to_json(segments, json_path):
        if Path(json_path).exists():
            with open(json_path, "r+", encoding="utf-8") as f:
                data = json.load(f)
                data.extend(segments)
                f.seek(0)
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.truncate()
        else:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(segments, f, ensure_ascii=False, indent=4)

    def create_dataset(self, audio_files, subtitle_files):
        """Create a Hugging Face dataset from a batch of files."""
        return IterableDataset.from_generator(
            self._generate_dataset_format(audio_files, subtitle_files)
        )

    def _generate_dataset_format(self, audio_files, subtitle_files):
        def generator():
            for audio_file, subtitle_file in zip(audio_files, subtitle_files):
                audio_array, sampling_rate = librosa.load(
                    f"{audio_file}.wav", sr=None, mono=True, dtype=np.float32
                )

                with open(subtitle_file, "r", encoding="utf-8") as f:
                    subtitle = json.load(f)

                for entry_idx, entry in enumerate(subtitle):
                    file_name = os.path.basename(audio_file)
                    yield {
                        "client_id": f"{file_name}_{entry_idx}",
                        "path": f"{audio_file}.wav",
                        "sentence": entry["text"],
                        "start": entry["start"],
                        "end": entry["start"] + entry["duration"],
                        "duration": entry["duration"],
                    }
                del audio_array
                gc.collect()

        return generator

    @staticmethod
    def iterable_to_dataset(iterable_dataset):
        """Convert an IterableDataset to a regular Dataset."""
        data_dict = defaultdict(list)
        for item in iterable_dataset:
            for key, value in item.items():
                data_dict[key].append(value)
        return Dataset.from_dict(data_dict)

    @staticmethod
    def create_dataset_from_json(json_file):
        """Create a Hugging Face dataset from a JSON file."""
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return Dataset.from_dict(
            {
                "client_id": [
                    f"{os.path.basename(item['audio_path'])}"
                    for i, item in enumerate(data)
                ],
                "path": [item["audio_path"] for item in data],
                "sentence": [item["text"] for item in data],
                "start": [item["start"] for item in data],
                "end": [item["end"] for item in data],
                "duration": [item["end"] - item["start"] for item in data],
            }
        )

    @staticmethod
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

    def process_and_create_dataset(self, json_path):
        """Process the crawled data and create a dataset."""
        dataset = self.create_dataset_from_json(json_path)
        return dataset

    def save_dataset(self, dataset, output_file):
        """Save the dataset to a JSON file."""
        self.convert_dataset_to_json(dataset, output_file)


def main():
    parser = HfArgumentParser(CrawlerArgs)
    args = parser.parse_args_into_dataclasses()[0]
    crawler = YoutubeCrawler(args)
    crawler.crawl()


if __name__ == "__main__":
    main()
