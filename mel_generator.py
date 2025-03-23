import torch
import torchaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class MelSpectrogramGenerator:
    def __init__(self, 
                 sample_rate: int = 22050,
                 n_mels: int = 80,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 win_length: int = 1024):
        
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        
        # Initialize mel spectrogram transformer
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            normalized=True
        )
        
    def generate_mel_spectrogram(self, audio_path: str, save_plot: bool = True) -> torch.Tensor:
        """Generate mel spectrogram from audio file"""
        try:
            # Load audio
            waveform, sr = torchaudio.load(audio_path)
            
            # Resample if necessary
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            
            # Generate mel spectrogram
            mel_spec = self.mel_transform(waveform)
            
            # Convert to log scale
            mel_spec = torch.log(mel_spec + 1e-9)
            
            # Save visualization if requested
            if save_plot:
                plot_path = Path(audio_path).with_suffix('.png')
                self._save_spectrogram_plot(mel_spec, plot_path)
            
            return mel_spec
            
        except Exception as e:
            print(f"Error generating mel spectrogram: {str(e)}")
            return None
    
    def _save_spectrogram_plot(self, mel_spec: torch.Tensor, save_path: str):
        """Save mel spectrogram visualization"""
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spec[0].numpy(), 
                  aspect='auto', 
                  origin='lower',
                  interpolation='none')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.ylabel('Mel Bands')
        plt.xlabel('Time')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close() 