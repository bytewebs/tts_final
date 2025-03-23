import torch
import numpy as np
import soundfile as sf
import os
from pathlib import Path
import matplotlib.pyplot as plt
from .model import DeepVoice3Model
from .config import DeepVoiceConfig
from .utils import VoiceUtils

class VoiceGenerator:
    def __init__(self, model_path: str, config_path: str):
        self.config = DeepVoiceConfig.from_json(config_path)
        self.model = DeepVoice3Model(self.config)
        self.model.build_model()
        
        # Load model weights if they exist
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            if checkpoint.get('model_state_dict'):
                self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.utils = VoiceUtils()
        
    def generate_speech(self, 
                       text: str, 
                       output_path: str, 
                       speaker_id: int = 0) -> dict:
        """
        Generate speech from text and save to file
        Returns metadata about the generation
        """
        # Process text
        text_sequence, durations = self.utils.process_text(text)
        text_tensor = torch.from_numpy(text_sequence).unsqueeze(0)
        
        # Generate audio
        with torch.no_grad():
            output, attention, mel = self.model.forward(text_tensor, speaker_id)
        
        # Create dummy audio for demonstration
        duration = len(text) * 0.1  # rough estimate of duration
        waveform = self.utils.create_dummy_audio(duration)
        
        # Save the audio file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(output_path, waveform, self.config.sample_rate)
        
        # Generate visualization
        fig_path = output_path.with_suffix('.png')
        self.utils.visualize_alignment(
            attention.numpy()[0],
            mel.numpy()[0],
            self.config.sample_rate,
            self.config.hop_size
        )
        plt.savefig(fig_path)
        plt.close()
        
        return {
            'audio_path': str(output_path),
            'visualization_path': str(fig_path),
            'duration': len(waveform) / self.config.sample_rate,
            'text': text,
            'speaker_id': speaker_id
        }
    
    def generate_samples(self, 
                        output_dir: str, 
                        num_samples: int = 5) -> list:
        """Generate multiple sample outputs"""
        sample_texts = [
            "This is a test of the voice generation system.",
            "The weather is beautiful today.",
            "Artificial intelligence is transforming the world.",
            "I hope this demonstration is helpful.",
            "Voice cloning technology is fascinating."
        ]
        
        results = []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(min(num_samples, len(sample_texts))):
            output_path = output_dir / f"sample_{i+1}.wav"
            result = self.generate_speech(
                text=sample_texts[i],
                output_path=str(output_path),
                speaker_id=i % 3  # Cycle through 3 speaker IDs
            )
            results.append(result)
        
        return results

def main():
    """Example usage of the generator"""
    import json
    
    # Setup paths
    model_path = "checkpoints/latest.pt"
    config_path = "config/voice_config.json"
    output_dir = "generated_samples"
    
    # Initialize generator
    generator = VoiceGenerator(model_path, config_path)
    
    # Generate samples
    results = generator.generate_samples(output_dir)
    
    # Save generation metadata
    with open(f"{output_dir}/metadata.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Generated {len(results)} samples in {output_dir}")
    for result in results:
        print(f"- {result['text']} -> {result['audio_path']}")

if __name__ == "__main__":
    main() 