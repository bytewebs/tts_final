import os
import pandas as pd
import shutil
import torch
from TTS.api import TTS
from training import train_voice_clone_model, DeepVoiceConfig
import librosa
import matplotlib
import numpy

# Step 1: Set up paths
DATASET_PATH = "./dataset"
WAVS_PATH = os.path.join(DATASET_PATH, "wavs")
METADATA_FILE = os.path.join(DATASET_PATH, "metadata.csv")
OUTPUT_PATH = "./output"

# Create necessary directories
os.makedirs(WAVS_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Step 2: Move audio files to the correct location
SOURCE_WAVS = "./wavs"
if os.path.exists(SOURCE_WAVS):
    for file in os.listdir(SOURCE_WAVS):
        if file.endswith(".wav"):
            shutil.copy2(os.path.join(SOURCE_WAVS, file), WAVS_PATH)

# Step 3: Create metadata.csv from list.txt
metadata = []
with open("list.txt", "r", encoding="utf-8") as f:
    for line in f:
        parts = line.strip().split("|")
        if len(parts) == 2:
            # Extract just the filename without the full path
            filename = os.path.basename(parts[0])
            metadata.append([filename, parts[1]])

df = pd.DataFrame(metadata, columns=["filename", "transcription"])
df.to_csv(METADATA_FILE, sep="|", index=False, header=False)

# Step 4: Initialize TTS with XTTS v2
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Step 5: Generate speech samples using voice cloning
print("Starting voice cloning process...")

# Generate test samples using voice cloning
test_texts = [
    "Hello, this is my trained TTS model speaking.",
    "The weather is beautiful today.",
    "I love learning about artificial intelligence."
]

# Use the first WAV file as the reference speaker
reference_speaker = os.path.join(WAVS_PATH, "1.wav")

for i, text in enumerate(test_texts):
    output_path = os.path.join(OUTPUT_PATH, f"generated_{i+1}.wav")
    tts.tts_to_file(
        text=text,
        speaker_wav=reference_speaker,
        language="en",
        file_path=output_path
    )
    print(f"Generated speech saved to: {output_path}")

print("Voice cloning completed! You can find the generated samples in the output directory.")
