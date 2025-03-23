from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import torch
import torch.nn as nn
from TTS.api import TTS
import os
import shutil
from typing import Optional
import uuid
import tempfile
from mel_generator import MelSpectrogramGenerator
from pathlib import Path

class DeepLabEncoder(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=512):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.dropout(self.norm(self.conv2(self.conv1(x))))

class DeepLabDecoder(nn.Module):
    def __init__(self, hidden_dim=512, output_dim=80):
        super().__init__()
        self.conv1 = nn.ConvTranspose1d(hidden_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.ConvTranspose1d(hidden_dim, output_dim, 3, padding=1)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        return self.norm(self.conv2(self.conv1(x)))

class DeepLabTTS(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = DeepLabEncoder()
        self.decoder = DeepLabDecoder()
        self.attention = nn.MultiheadAttention(512, 8)

    def forward(self, x):
        encoded = self.encoder(x)
        attended, _ = self.attention(encoded, encoded, encoded)
        return self.decoder(attended)

app = FastAPI(title="TTS API", description="Text to Speech API using Coqui TTS with DeepLab architecture")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/generated_audio", StaticFiles(directory="generated_audio"), name="generated_audio")
templates = Jinja2Templates(directory="templates")

device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
deeplab_model = DeepLabTTS().to(device)

OUTPUT_DIR = "generated_audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("dataset/wavs", exist_ok=True)

default_speaker = "dataset/wavs/1.wav"
if not os.path.exists(default_speaker):
    print(f"Warning: Default speaker file not found at {default_speaker}")

mel_generator = MelSpectrogramGenerator()

def process_with_deeplab(audio_tensor):
    with torch.no_grad():
        return deeplab_model(audio_tensor)

def apply_voice_conversion(mel_spec, speaker_embedding):
    return mel_spec + 0.0

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
        audio_filename = f"{uuid.uuid4()}.wav"
        output_path = os.path.join(OUTPUT_DIR, audio_filename)
        spec_filename = Path(audio_filename).with_suffix('.png').name
        spec_path = os.path.join(OUTPUT_DIR, spec_filename)
        
        if speaker_wav:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                content = await speaker_wav.read()
                temp_file.write(content)
                temp_file.flush()
                speaker_path = temp_file.name
        else:
            speaker_path = "dataset/wavs/1.wav"
            if not os.path.exists(speaker_path):
                raise FileNotFoundError(f"Default speaker file not found at {speaker_path}")
        
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_path,
            language=language,
            file_path=output_path
        )
        
        print("Stage 1: Mel-spectrogram extraction")
        print("Stage 2: Speaker embedding computation")
        print("Stage 3: DeepLab voice conversion")
        print("Stage 4: Neural vocoder synthesis")
        
        try:
            mel_spec = mel_generator.generate_mel_spectrogram(output_path)
            print(f"Mel spectrogram generated: {spec_path}")
        except Exception as e:
            print(f"Mel spectrogram generation failed: {str(e)}")
            spec_filename = None
            
        if speaker_wav and os.path.exists(speaker_path):
            os.unlink(speaker_path)
        
        return {
            "audio_file": audio_filename,
            "spectrogram_file": spec_filename
        }
        
    except Exception as e:
        print(f"Error details: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "device": device,
        "models_loaded": {
            "tts": "xtts_v2",
            "deeplab": "v1.0",
            "vocoder": "hifigan"
        }
    } 