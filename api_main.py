# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import torch
from src.inference.flexible_inference import ChineseTaiwaneseASRInference
import librosa
import io
import os
from typing import Optional, List
from uuid import UUID, uuid4
from datetime import datetime
from src.utils.logging import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

model_name = os.getenv("MODEL_PATH", "openai/whisper-small")

# Initialize ASR model
try:
    logger.debug("Attempting to initialize ASR model")
    model = ChineseTaiwaneseASRInference(
        model_path=model_name,
        device=(
            "cuda"
            if torch.cuda.is_available()
            and os.getenv("USE_GPU", "False").lower() == "true"
            else "cpu"
        ),
        language=os.getenv("LANGUAGE", "chinese"),
    )
    logger.info("ASR model initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize ASR model: {str(e)}")
    raise


class TranscriptionRequest(BaseModel):
    audio_file: UploadFile
    max_alternatives: Optional[int] = 1


class TranscriptionResponse(BaseModel):
    id: UUID
    text: str
    confidence: float
    timestamp: datetime


# Simulating a database
transcriptions_db = {}


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        logger.debug("Reading file contents")
        contents = await file.read()

        logger.debug("Loading audio with librosa")
        audio, sr = librosa.load(io.BytesIO(contents), sr=None)

        # Use your ASR model for transcription
        transcription = model.transcribe_batch(audio, sr)[0]

        # Create and store the transcription result
        transcription_id = uuid4()
        result = TranscriptionResponse(
            id=transcription_id,
            text=transcription,
            confidence=0.95,  # Assuming a fixed confidence for simplicity
            timestamp=datetime.now(),
        )
        transcriptions_db[transcription_id] = result

        return result
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise HTTPException(
            status_code=500, detail="An error occurred during transcription"
        )


@app.get("/transcriptions", response_model=List[TranscriptionResponse])
async def list_transcriptions():
    return list(transcriptions_db.values())


@app.get("/transcription/{transcription_id}", response_model=TranscriptionResponse)
async def get_transcription(transcription_id: UUID):
    if transcription_id not in transcriptions_db:
        raise HTTPException(status_code=404, detail="Transcription not found")
    return transcriptions_db[transcription_id]


@app.delete("/transcription/{transcription_id}")
async def delete_transcription(transcription_id: UUID):
    if transcription_id not in transcriptions_db:
        raise HTTPException(status_code=404, detail="Transcription not found")
    del transcriptions_db[transcription_id]
    return {"message": "Transcription deleted successfully"}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


# if __name__ == "__main__":
#     import uvicorn

#     logger.info("Starting FastAPI application on port %s", os.getenv("PORT", 8000))
#     uvicorn.run(app, host="0.0.0.0", port=8000))
