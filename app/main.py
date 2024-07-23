# main.py
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import torch
from src.inference.flexible_inference import ChineseTaiwaneseASRInference
import librosa
import io

app = FastAPI()

# Initialize ASR model
model = ChineseTaiwaneseASRInference(
    model_path="path/to/your/model",
    device="cuda" if torch.cuda.is_available() else "cpu",
    language="chinese"
)


class TranscriptionResponse(BaseModel):
    transcription: str


@app.post("/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    contents = await file.read()
    audio, sr = librosa.load(io.BytesIO(contents), sr=None)
    
    # Use your ASR model for transcription
    transcription = model.transcribe_batch(audio, sr)[0]
    
    return TranscriptionResponse(transcription=transcription)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)