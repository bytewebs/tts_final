import librosa
import numpy as np
import torch
from typing import Tuple
import matplotlib.pyplot as plt

class VoiceUtils:
    @staticmethod
    def visualize_alignment(alignment: np.ndarray, 
                          spectrogram: np.ndarray, 
                          sample_rate: int, 
                          hop_length: int) -> None:
        """Visualize alignment and spectrogram"""
        plt.figure(figsize=(16, 16))
        
        # Plot alignment
        plt.subplot(2, 1, 1)
        plt.imshow(alignment.T, aspect="auto", origin="lower", interpolation=None)
        plt.xlabel("Decoder timestamp")
        plt.ylabel("Encoder timestamp")
        plt.colorbar()
        
        # Plot spectrogram
        plt.subplot(2, 1, 2)
        librosa.display.specshow(
            spectrogram.T,
            sr=sample_rate,
            hop_length=hop_length,
            x_axis="time",
            y_axis="linear"
        )
        plt.xlabel("Time")
        plt.ylabel("Hz")
        plt.tight_layout()
        plt.colorbar()
    
    @staticmethod
    def process_text(text: str) -> Tuple[np.ndarray, np.ndarray]:
        """Process input text for model"""
        # Convert text to sequence of indices
        char_to_ind = {char: i for i, char in enumerate(set(text))}
        text_sequence = np.array([char_to_ind[c] for c in text])
        
        # Create dummy duration sequence
        durations = np.ones_like(text_sequence) * 8
        
        return text_sequence, durations
    
    @staticmethod
    def create_mel_spectrogram(audio: np.ndarray, 
                             sample_rate: int,
                             n_mels: int = 80) -> np.ndarray:
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_mels=n_mels,
            n_fft=2048,
            hop_length=512
        )
        return librosa.power_to_db(mel_spec, ref=np.max)
    
    @staticmethod
    def griffin_lim(spectrogram: np.ndarray,
                    n_iter: int = 50,
                    window: str = 'hann',
                    n_fft: int = 2048,
                    hop_length: int = 512) -> np.ndarray:
        angles = np.exp(2j * np.pi * np.random.rand(*spectrogram.shape))
        S = np.abs(spectrogram)
        y = librosa.istft(S * angles, hop_length=hop_length, window=window)
        
        for _ in range(n_iter):
            D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window)
            angles = np.exp(1j * np.angle(D))
            y = librosa.istft(S * angles, hop_length=hop_length, window=window)
            
        return y
    
    @staticmethod
    def create_dummy_audio(duration_seconds: float, sample_rate: int = 22050) -> np.ndarray:
        """Create dummy audio signal for demonstration"""
        # Generate a simple sine wave with varying frequency
        t = np.linspace(0, duration_seconds, int(duration_seconds * sample_rate))
        frequencies = [440, 880, 1320]  # A4, A5, E6
        
        # Combine multiple frequencies with different amplitudes
        audio = np.zeros_like(t)
        for i, f in enumerate(frequencies):
            amplitude = 0.5 / (i + 1)
            audio += amplitude * np.sin(2 * np.pi * f * t)
        
        # Add some noise
        noise = np.random.randn(len(t)) * 0.01
        audio = audio + noise
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        return audio 