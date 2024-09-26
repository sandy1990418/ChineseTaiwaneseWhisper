from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict
import librosa
import json
from src.utils.logging import logger
import os
from .audio_saver import JsonAppendSaver, SoundfileSaver


class AudioProcessStrategy(ABC):
    @abstractmethod
    def process(
        self,
        audio_file: Path,
        subtitle_file: Path,
        json_path: str,
        max_duration: float,
    ) -> List[Dict]:
        raise NotImplementedError("This method must be implemented.")


class LibrosaAudioProcessStrategy(AudioProcessStrategy):
    def __init__(self, output_dir: Path, file_prefix: str):
        self.output_dir = output_dir
        self.file_prefix = file_prefix
        self.json_saver = JsonAppendSaver()
        self.soundfile_saver = SoundfileSaver()

    def process(
        self, audio_file: Path, subtitle_file: Path, json_path: str, max_duration: float
    ) -> List[Dict]:
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

            if end_time - current_start >= max_duration:
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
            self.json_saver.save(segments, json_path)
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
        os.makedirs(os.path.dirname(segment_path), exist_ok=True)
        self.soundfile_saver.save(split_audio, sr, segment_path)
        # sf.write(segment_path, split_audio, sr)

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


class AudioProcessStrategyFactory:
    @staticmethod
    def create_process_strategy(
        output_dir: str, file_prefix: str, strategy_type: str = "librosa"
    ) -> AudioProcessStrategy:
        if strategy_type == "librosa":
            return LibrosaAudioProcessStrategy(output_dir, file_prefix)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")


class AudioProcessor:
    def __init__(self, strategy: AudioProcessStrategy):
        self.strategy = strategy

    def process_audio(
        self,
        audio_file: Path,
        subtitle_file: Path,
        json_path: str,
        max_duration: float,
    ) -> List[Dict]:
        return self.strategy.process(audio_file, subtitle_file, json_path, max_duration)