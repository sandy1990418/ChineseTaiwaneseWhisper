from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import yt_dlp
import requests
import os
import subprocess
from src.utils.logging import logger


class AudioConverter:
    @staticmethod
    def convert(audio_file: Path, audio_type: str = "mp3"):
        """
        Convert audio file to wav.

        Args:
            audio_file (Path): Path to the audio file.
            audio_type (str): Audio type.
        """
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
                str(audio_file.with_suffix(f".{audio_type}"))
            )  # Remove the original mp3 file
        except subprocess.CalledProcessError as e:
            logger.error(f"Error converting mp3 to wav: {e}")
            return None

        return wav_file


#  abstract Download Strategy class
class AudioDownloadStrategy(ABC):
    def __init__(self, audio_converter: AudioConverter):
        self.audio_converter = audio_converter

    @abstractmethod
    def download(self, video_id: str, audio_file: Path):
        return NotImplementedError("Download method must be implemented")

    def download_and_convert(self, video_id: str, audio_file: Path, audio_type: str = "mp3"):
        audio_file = self.download(video_id, audio_file)
        audio_file = self.audio_converter.convert(audio_file, audio_type)
        return audio_file


class YTDLPDownloadStrategy(AudioDownloadStrategy):
    def __init__(self, audio_converter: AudioConverter, ffmpeg_path: Optional[str] = None):
        super().__init__(audio_converter)
        self.ffmpeg_path = ffmpeg_path

    def download(self, video_id: str, audio_file: Path):
        """
        Download audio with yt-dlp. It may be prevented by youtube. If it is, try to download with cobalt.

        Args:
            video_id (str): YouTube video ID.
            audio_file (Path): Path to save the audio file.

        Returns:
            Path: Path to the downloaded audio file.
        """
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


class CobaltDownloadStrategy(AudioDownloadStrategy):
    def __init__(self, audio_converter: AudioConverter):
        super().__init__(audio_converter)

    def download(self, video_id: str, audio_file: Path):
        """
        Download audio with cobalt.
        Thanks to Cobalt! Your work is truly great.
        https://github.com/imputnet/cobalt

        Args:
            video_id (str): YouTube video ID.
            audio_file (Path): Path to save the audio file.

        Returns:
            Path: Path to the downloaded audio file.
        """
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

        response = requests.post(url, json=params, headers=headers)
        if response.status_code == 200:
            result = response.json()
            download_url = result["url"]
            # Step 2: Download the audio content from the stream
            logger.info("Start to stream download using Cobalt API.")
            with requests.get(download_url, stream=True) as stream_response:
                stream_response.raise_for_status()
                with open(audio_file, "wb") as file:
                    for chunk in stream_response.iter_content(chunk_size=8192):
                        file.write(chunk)

            logger.info("Download successful!")
            return audio_file
        else:
            raise Exception(
                f"Failed to download audio with Cobalt API. Status code: {response.status_code}"
            )


class FallbackDownloadStrategy(AudioDownloadStrategy):
    """
    Download strategy that tries to download with the primary strategy first.
    If it fails, it tries to download with the secondary strategy.
    """

    def __init__(
        self,
        primary_strategy: AudioDownloadStrategy,
        secondary_strategy: AudioDownloadStrategy,
    ):
        super().__init__(primary_strategy.audio_converter)
        self.primary_strategy = primary_strategy
        self.secondary_strategy = secondary_strategy

    def download(self, video_id: str, audio_file: Path) -> Path:
        try:
            return self.primary_strategy.download(video_id, audio_file)
        except Exception as e:
            logger.error(
                f"Failed to download audio with {self.primary_strategy.__class__.__name__}. Error: {e}"
            )
            return self.secondary_strategy.download(video_id, audio_file)


class AudioDownloadStrategyFactory:
    """
    Factory to create a download strategy.
    """

    @staticmethod
    def create_download_strategy(ffmpeg_path: Optional[str] = None):
        audio_converter = AudioConverter()
        return FallbackDownloadStrategy(
            primary_strategy=YTDLPDownloadStrategy(audio_converter, ffmpeg_path),
            secondary_strategy=CobaltDownloadStrategy(audio_converter),
        )