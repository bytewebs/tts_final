import torch
from TTS.api import TTS
import os

def generate_speech(text, output_path, speaker_wav, language="en"):
    """
    Generate speech from text using XTTS v2 voice cloning
    """
    # Get device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize TTS with XTTS v2
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    
    # Generate speech
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language=language,
        file_path=output_path
    )
    print(f"Generated speech saved to: {output_path}")

if __name__ == "__main__":
    # Example usage
    OUTPUT_DIR = "generated_speech"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Reference speaker WAV file
    SPEAKER_WAV = "./dataset/wavs/1.wav"
    
    # Example text to synthesize
    test_texts = [
        "Hello, this is my trained TTS model speaking.",
        "The weather is beautiful today.",
        "I love learning about artificial intelligence."
    ]
    
    for i, text in enumerate(test_texts):
        output_path = os.path.join(OUTPUT_DIR, f"generated_{i+1}.wav")
        generate_speech(text, output_path, SPEAKER_WAV) 