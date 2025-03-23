import torch
import torchaudio
import librosa
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

class AudioProcessor:
    def __init__(self, 
                 sample_rate: int = 22050,
                 n_mels: int = 80,
                 n_fft: int = 1024,
                 hop_length: int = 256,
                 win_length: int = 1024,
                 f_min: int = 0,
                 f_max: Optional[int] = 8000):
        
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max if f_max else sample_rate // 2
        
        # Initialize mel filterbank
        self.mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=f_min,
            fmax=f_max
        )
        self.mel_basis = torch.FloatTensor(self.mel_basis)
        
        # Initialize window function
        self.window = torch.hann_window(win_length)
        
    def load_audio(self, file_path: str) -> Tuple[torch.Tensor, int]:
        """Load and preprocess audio file"""
        try:
            # Load audio
            waveform, sr = torchaudio.load(file_path)
            
            # Convert to mono if stereo
            if waveform.size(0) > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
            
            # Normalize
            waveform = waveform / torch.abs(waveform).max()
            
            return waveform, self.sample_rate
            
        except Exception as e:
            raise RuntimeError(f"Error loading audio file: {str(e)}")
    
    def get_mel_spectrogram(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to mel spectrogram"""
        # Ensure input is 2D
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Compute STFT
        D = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True
        )
        
        # Convert to power spectrogram
        magnitudes = torch.abs(D) ** 2
        
        # Apply mel filterbank
        mel_spec = torch.matmul(self.mel_basis, magnitudes)
        
        # Convert to log scale
        log_mel_spec = torch.log10(torch.clamp(mel_spec, min=1e-5))
        
        # Normalize
        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / log_mel_spec.std()
        
        return log_mel_spec
    
    def save_mel_spectrogram(self, mel_spec: torch.Tensor, file_path: str):
        """Save mel spectrogram visualization"""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spec.squeeze().numpy(), 
                  aspect='auto', 
                  origin='lower',
                  interpolation='none')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-frequency spectrogram')
        plt.ylabel('Mel filter')
        plt.xlabel('Time')
        plt.tight_layout()
        plt.savefig(file_path)
        plt.close() 