from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import torch
from TTS.api import TTS
import os
import shutil
from typing import Optional
import uuid
import tempfile

app = FastAPI(title="TTS API", description="Text to Speech API using Coqui TTS")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize TTS
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Create output directory
OUTPUT_DIR = "generated_audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate-speech")
async def generate_speech(
    text: str = Form(...),
    language: str = Form("en"),
    speaker_wav: Optional[UploadFile] = File(None)
):
    try:
        # Generate unique filename
        filename = f"{uuid.uuid4()}.wav"
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        # Handle speaker file
        if speaker_wav:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                content = await speaker_wav.read()
                temp_file.write(content)
                temp_file.flush()
                speaker_path = temp_file.name
        else:
            # Use default speaker
            speaker_path = "dataset/wavs/1.wav"
        
        # Generate speech
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_path,
            language=language,
            file_path=output_path
        )
        
        # Clean up temporary file if it exists
        if speaker_wav and os.path.exists(speaker_path):
            os.unlink(speaker_path)
        
        return FileResponse(
            output_path,
            media_type="audio/wav",
            filename=filename
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": device} 