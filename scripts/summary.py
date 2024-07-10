from openai import OpenAI
from typing import List, Dict
import json
import os 
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI KPI
with open('api_key/key.txt', 'r') as f: 
    api_key = f.read()


def read_transcript_json(file_path: str) -> List[Dict]:
    """
    Read the transcript JSON file and return its content.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def extract_text_from_transcript(transcript: Dict) -> str:
    """
    Extract the text content from the transcript, joining transcriptions from all channels.
    """
    all_transcriptions = []
    for item in transcript:
        if 'channels' in item:
            for channel in item['channels']:
                transcriptions = channel.get('transcriptions', '')
                if isinstance(transcriptions, str):
                    all_transcriptions.append(transcriptions)
                elif isinstance(transcriptions, list):
                    all_transcriptions.extend(transcriptions)
        elif 'message' in item:
            transcriptions = item['message']
            timestamp = item.get('timestamp', None)
            all_transcriptions.append(f"[{timestamp}] {transcriptions}")
        
    # Join all transcriptions, removing any empty strings
    return ' '.join(filter(bool, all_transcriptions))


def summarize_transcript(transcript: str) -> str:
    """
    Send the transcript to OpenAI's API for summarization.
    """
    client = OpenAI(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": """請將以下會議內容紀錄並彙整為清晰、\
                完整的會議紀要，並以繁體中文返回。請運用 Chain of Thought（CoT）方法，\
                逐步分析和總結每個議題。請確保包含以下要點：\
                主要討論議題和決策
                任何待辦事項或後續步驟
                重要意見和建議
                務必確保會議紀要簡潔明瞭且不遺漏任何重要信息。\
                最後以Markdown Code的形式提供，不需要會議時間
                """},
                {"role": "user", "content": f"Please summarize the following transcript:\n\n{transcript}"}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"An error occurred while summarizing: {e}")
        return None


def process_transcript_file(file_path: str) -> str:
    """
    Process a single transcript file: read, extract text, and summarize.
    """
    transcript_data = read_transcript_json(file_path)
    transcript_text = extract_text_from_transcript(transcript_data)
    return summarize_transcript(transcript_text)

# TODO: put summary to gradio_interface


def main():
    # Directory containing transcript JSON files
    transcript_dir = "asr_transcription_streaming_cache"  # "output"
    
    # Process all JSON files in the directory
    for filename in os.listdir(transcript_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(transcript_dir, filename)
            logger.info(f"Processing file: {filename}")
            
            summary = process_transcript_file(file_path)

            if summary:
                logger.info(f"Summary for {filename}:")
                logger.info(summary)
                logger.info("\n" + "="*50 + "\n")
            else:
                logger.error(f"Failed to generate summary for {filename}")


if __name__ == "__main__":
    main()