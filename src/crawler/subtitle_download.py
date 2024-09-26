from abc import ABC, abstractmethod
from pathlib import Path
import json
from youtube_transcript_api import YouTubeTranscriptApi
from src.utils import logger
from typing import List, Dict
import openai
from pydub import AudioSegment
import os
from dotenv import load_dotenv

load_dotenv()


class SubtitleDownloadStrategy(ABC):
    @abstractmethod
    def download(self, video_id: str, subtitle_file: Path):
        return NotImplementedError(
            "SubtitleDownloadStrategy must implement download method"
        )


class YouTubeTranscriptDownloadStrategy(SubtitleDownloadStrategy):
    def download(
        self,
        video_id: str,
        subtitle_file: Path,
        languages: List[str] = ["zh-TW", "zh-CN", "zh"],
    ):
        
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
        with open(str(subtitle_file), "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False, indent=2)


MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB in bytes
INITIAL_CHUNK_DURATION = 10 * 60 * 1000  # 10 minutes in milliseconds


class OpenAISubtitleDownloadStrategy(SubtitleDownloadStrategy):
    def __init__(self):
        logger.info("Initializing OpenAISubtitleDownloadStrategy, loading OpenAI client")
        self.client = openai.OpenAI()

    def download(self, video_info: Dict[str, str], audio_file: str, subtitle_file: Path):
        try:
            result = self._transcribe_audio(video_info, audio_file)
            with open(str(subtitle_file), "w", encoding="utf-8") as f:
                json.dump(result["transcript"], f, ensure_ascii=False, indent=2)
            return subtitle_file
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return None

    def _transcribe_audio(self, video_info: Dict[str, str], audio_file: str):
        try:
            output_dir = '/'.join(audio_file.split("/")[:-1])
            audio = AudioSegment.from_wav(audio_file)
            transcripts = []
            start = 0
            
            while start < len(audio):
                end = min(start + INITIAL_CHUNK_DURATION, len(audio))

                while True:
                    chunk = audio[start:end]
                    chunk_file = os.path.join(output_dir, "temp_chunk.mp3")
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


class FallbackSubtitleDownloadStrategy(SubtitleDownloadStrategy):
    def __init__(self, primary_strategy: SubtitleDownloadStrategy, secondary_strategy: SubtitleDownloadStrategy):
        self.primary_strategy = primary_strategy
        self.secondary_strategy = secondary_strategy

    def download(self, video_id: str, subtitle_file: Path, audio_file: Path):
        try:
            return self.primary_strategy.download(video_id, subtitle_file)
        except Exception as e:
            logger.error(f"Failed to download subtitles with {self.primary_strategy.__class__.__name__}. Error: {e}")
            logger.info(f"Fallback to {self.secondary_strategy.__class__.__name__}")
            
            if isinstance(self.secondary_strategy, OpenAISubtitleDownloadStrategy):
                video_info = {
                    "video_id": video_id,
                    "video_title": f"video_{video_id}",
                    "channel_title": "unknown_channel",
                }
                
                return self.secondary_strategy.download(video_info, str(audio_file), str(subtitle_file))


class SubtitleDownloadStrategyFactory:
    @staticmethod
    def create_subtitle_download_strategy() -> SubtitleDownloadStrategy:
        return FallbackSubtitleDownloadStrategy(
            YouTubeTranscriptDownloadStrategy(), OpenAISubtitleDownloadStrategy()
        )