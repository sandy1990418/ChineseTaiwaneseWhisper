from pathlib import Path
from typing import Optional
from src.config import CrawlerArgs
from src.utils.logging import logger
from src.crawler.audio_download import AudioDownloadStrategyFactory
from src.crawler.subtitle_download import SubtitleDownloadStrategyFactory
from src.crawler.audio_process import AudioProcessStrategyFactory, AudioProcessor
from src.crawler.audio_saver import HuggingFaceDatasetSaver
import re
import yt_dlp
import subprocess
import os 


class YoutubeCrawler:
    def __init__(self, args: CrawlerArgs):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.ffmpeg_path = args.ffmpeg_path
        self.file_prefix = args.file_prefix
        self.batch_size = args.batch_size
        self.max_duration = args.max_duration
        self.ffmpeg_path = args.ffmpeg_path

        self._create_output_dirs()

        self.audio_download_strategy = (
            AudioDownloadStrategyFactory.create_download_strategy(self.ffmpeg_path)
        )
        self.subtitle_download_strategy = (
            SubtitleDownloadStrategyFactory.create_subtitle_download_strategy()
        )
        self.audio_process_strategy = (
            AudioProcessStrategyFactory.create_process_strategy(
                args.output_dir, args.file_prefix
            )
        )
        self.audio_processor = AudioProcessor(self.audio_process_strategy)
        self.dataset_saver = HuggingFaceDatasetSaver()

    def crawl(self):
        self._check_ffmpeg()
        json_path = (
            self.output_dir / self.file_prefix / f"{self.args.dataset_name}.json"
        )

        if self.args.playlist_urls:
            logger.info(f"Processing YouTube playlists: {self.args.playlist_urls}")
            for play_idx, playlist_url in enumerate(self.args.playlist_urls):
                self._process_youtube_playlist(play_idx, playlist_url, json_path)

        if self.args.audio_dir:
            logger.info(f"Processing existing audio files from: {self.args.audio_dir}")
            self._process_existing_audio_files(self.args.audio_dir, json_path)

        logger.info(f"All segments saved to: {json_path}")

    def _create_output_dirs(self):
        self.audio_dir = os.path.join(self.output_dir, self.file_prefix, "audio")
        self.subtitle_dir = os.path.join(self.output_dir, self.file_prefix, "subtitles")
        os.makedirs(self.audio_dir, exist_ok=True)
        os.makedirs(self.subtitle_dir, exist_ok=True)

    def _check_ffmpeg(self):
        """Check if FFmpeg is installed and accessible."""
        try:
            if self.ffmpeg_path:
                subprocess.run(
                    [self.ffmpeg_path, "-version"],
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

    def _extract_playlist_id(self, url: str) -> Optional[str]:
        playlist_id_match = re.search(r"(?:list=)([a-zA-Z0-9_-]+)", url)
        return playlist_id_match.group(1) if playlist_id_match else None

    def _process_youtube_playlist(
        self, play_idx: int, playlist_url: str, json_path: Path
    ):
        playlist_id = self._extract_playlist_id(playlist_url)
        if not playlist_id:
            logger.error(f"Invalid playlist URL: {playlist_url}")
            return

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
                self.audio_processor.process_audio(
                    audio_file, subtitle_file, json_path, self.max_duration
                )

            if index + 1 >= self.batch_size:
                logger.info(f"Reached batch size limit of {self.batch_size}. Stopping.")
                break

    def _download_youtube_audio_and_subtitles(
        self, video_id: str, play_idx: int, file_index: int
    ) -> Optional[tuple[Path, Path]]:
        file_name = f"{self.file_prefix}_{play_idx:04d}_{file_index:04d}"
        audio_file = Path(os.path.join(self.audio_dir, f"{file_name}.mp3"))
        subtitle_file = Path(os.path.join(self.subtitle_dir, f"{file_name}.json"))

        if not audio_file.exists():
            try:
                audio_file = self.audio_download_strategy.download_and_convert(video_id, audio_file)
            except Exception as e:
                logger.error(f"Error downloading audio for video {video_id}: {e}")
                return None
        if not subtitle_file.exists():
            try:
                
                self.subtitle_download_strategy.download(video_id, subtitle_file, audio_file)
            except Exception as e:
                logger.error(f"Error downloading subtitles for video {video_id}: {e}")
                return None

        return audio_file, subtitle_file

    def _process_existing_audio_files(self, audio_dir: Path, json_path: Path):
        audio_files = list(audio_dir.glob("*.wav"))
        for audio_file in audio_files:
            subtitle_file = audio_file.with_suffix(".json")
            if subtitle_file.exists():
                self.audio_processor.process_audio(
                    audio_file, subtitle_file, json_path, self.max_duration
                )
            else:
                logger.warning(
                    f"Subtitle file not found for {audio_file}. Skipping this audio file."
                )

    def process_and_create_dataset(self, json_path: Path):
        dataset = self.dataset_saver.create_dataset_from_json(json_path)
        return dataset

    def save_dataset(self, dataset, output_file: Path):
        self.dataset_saver.convert_dataset_to_json(dataset, output_file)


def main():
    from transformers import HfArgumentParser

    parser = HfArgumentParser(CrawlerArgs)
    args = parser.parse_args_into_dataclasses()[0]
    crawler = YoutubeCrawler(args)
    crawler.crawl()


if __name__ == "__main__":
    main()
