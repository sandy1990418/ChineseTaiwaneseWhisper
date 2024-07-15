from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class CrawlerArgs:
    # List of YouTube playlist URLs to crawl
    playlist_urls: List[str] = field(
        default_factory=lambda: [],
        metadata={"help": "YouTube playlist URLs to crawl"}
    )

    # Directory to save audio files and dataset
    output_dir: str = field(
        default="./output",
        metadata={"help": "Directory to save audio files and dataset"}
    )

    # Name of the output dataset file
    dataset_name: str = field(
        default="youtube_dataset",
        metadata={"help": "Name of the output dataset file"}
    )

    # Path to FFmpeg executable (optional)
    ffmpeg_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to FFmpeg executable"}
    )

    # Prefix for audio and subtitle files
    file_prefix: str = field(
        default="youtube",
        metadata={"help": "Prefix for audio and subtitle files"}
    )
    # Prefix for audio and subtitle files
    batch_size: int = field(
        default=20,
        metadata={"help": "Prefix for audio and subtitle files"}
    )