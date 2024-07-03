import os
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
from pydub import AudioSegment


def download_youtube_audio_and_subtitles(video_id, output_dir):
    """Download YouTube audio and subtitles for a given video ID."""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
    }

    # Download audio
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f'https://www.youtube.com/watch?v={video_id}'])

    # Get subtitles
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['zh-TW', 'zh-CN', 'zh'])
    except Exception as e:
        print(f"Error getting transcript for video {video_id}: {e}")
        return None

    # Save subtitles and create segmented audio
    subtitle_file = os.path.join(output_dir, f"{video_id}_subtitles.txt")
    with open(subtitle_file, 'w', encoding='utf-8') as f:
        for entry in transcript:
            start_time = entry['start']
            duration = entry['duration']
            text = entry['text']
            f.write(f"{start_time:.2f}\t{duration:.2f}\t{text}\n")

            # Segment audio
            audio = AudioSegment.from_wav(os.path.join(output_dir, f"{video_id}.wav"))
            segment = audio[int(start_time*1000):int((start_time+duration)*1000)]
            segment.export(os.path.join(output_dir, f"{video_id}_{start_time:.2f}.wav"), format="wav")

    return subtitle_file


def crawl_youtube_playlist(playlist_id, output_dir):
    """Crawl all videos in a YouTube playlist."""
    ydl_opts = {
        'extract_flat': True,
        'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        playlist_dict = ydl.extract_info(f'https://www.youtube.com/playlist?list={playlist_id}', download=False)

    for video in playlist_dict['entries']:
        video_id = video['id']
        print(f"Processing video: {video_id}")
        download_youtube_audio_and_subtitles(video_id, output_dir)


if __name__ == "__main__":
    playlist_id = "YOUR_PLAYLIST_ID"
    output_dir = "./youtube_data"
    os.makedirs(output_dir, exist_ok=True)
    crawl_youtube_playlist(playlist_id, output_dir)