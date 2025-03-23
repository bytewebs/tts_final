import os
import json
from pathlib import Path
from training.generate import VoiceGenerator
from training.train import train_voice_clone_model

# Create necessary directories
BASE_DIR = Path("voice_clone_project")
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
CONFIG_DIR = BASE_DIR / "config"
OUTPUT_DIR = BASE_DIR / "output"
SAMPLES_DIR = OUTPUT_DIR / "samples"

for directory in [CHECKPOINTS_DIR, CONFIG_DIR, OUTPUT_DIR, SAMPLES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Create a sample config file
config = {
    "sample_rate": 22050,
    "hop_size": 256,
    "n_mels": 80,
    "epochs": 5,
    "batch_size": 32,
    "learning_rate": 0.001,
    "fmin": 0,
    "fmax": 8000,
    "rescaling": False,
    "rescaling_max": 0.999,
    "allow_clipping_in_normalization": False,
    "builder": "deepvoice3_multispeaker"
}

# Save config
config_path = CONFIG_DIR / "voice_config.json"
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

def main():
    print("Starting Voice Cloning System")
    print("-" * 50)

    # 1. Train the model
    print("\n1. Training Model...")
    model = train_voice_clone_model(
        checkpoint_path=str(CHECKPOINTS_DIR / "latest.pt"),
        preset_path=str(config_path),
        data_dir=str(BASE_DIR / "data"),
        output_dir=str(CHECKPOINTS_DIR)
    )

    # 2. Initialize generator
    print("\n2. Initializing Generator...")
    generator = VoiceGenerator(
        model_path=str(CHECKPOINTS_DIR / "checkpoint_epoch_5.pt"),
        config_path=str(config_path)
    )

    # 3. Generate samples
    print("\n3. Generating Samples...")
    test_texts = [
        "This is a test of the voice generation system.",
        "The weather is beautiful today.",
        "Artificial intelligence is transforming the world.",
        "Voice cloning technology is fascinating.",
        "I hope this demonstration is helpful."
    ]

    for i, text in enumerate(test_texts, 1):
        print(f"\nGenerating sample {i}...")
        result = generator.generate_speech(
            text=text,
            output_path=str(SAMPLES_DIR / f"sample_{i}.wav"),
            speaker_id=i % 3
        )
        print(f"Generated: {result['audio_path']}")
        print(f"Visualization: {result['visualization_path']}")
        print(f"Duration: {result['duration']:.2f} seconds")

    print("\n4. Generating Multiple Samples at Once...")
    results = generator.generate_samples(
        output_dir=str(SAMPLES_DIR / "batch"),
        num_samples=3
    )

    print("\nProcess Complete!")
    print(f"Check the output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

# Initialize generator
generator = VoiceGenerator(
    model_path="voice_clone_project/checkpoints/latest.pt",
    config_path="voice_clone_project/config/voice_config.json"
)

# Generate a single sample
result = generator.generate_speech(
    text="This is a test.",
    output_path="voice_clone_project/output/custom_sample.wav",
    speaker_id=0
) 