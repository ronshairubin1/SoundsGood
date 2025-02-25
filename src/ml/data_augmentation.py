# SoundClassifier_v08/src/ml/data_augmentation.py

import numpy as np
import librosa
import logging
from .constants import SAMPLE_RATE

def time_shift(audio, shift_max=0.2):
    """Shift the audio by a random fraction of total length."""
    shift = np.random.randint(
        int(SAMPLE_RATE * -shift_max), 
        int(SAMPLE_RATE * shift_max)
    )
    return np.roll(audio, shift)

def change_pitch(audio, sr=SAMPLE_RATE, pitch_range=2.0):
    """Randomly shift pitch by ±pitch_range semitones."""
    n_steps = np.random.uniform(-pitch_range, pitch_range)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def change_speed(audio, speed_range=0.1):
    """Randomly time-stretch or compress by ±speed_range around 1.0."""
    speed_factor = np.random.uniform(1 - speed_range, 1 + speed_range)
    return librosa.effects.time_stretch(audio, rate=speed_factor)

def add_colored_noise(audio, noise_type='white', noise_factor=0.005):
    """
    Add colored noise to an audio signal.
    Args:
        noise_type: 'white', 'pink', or 'brown'
        noise_factor: scaling factor
    """
    if noise_type == 'white':
        noise = np.random.randn(len(audio))
    elif noise_type == 'pink':
        f = np.fft.fftfreq(len(audio))
        f = np.abs(f)
        f[0] = 1e-6  # Avoid division by zero
        pink = np.random.randn(len(audio)) / np.sqrt(f)
        noise = np.fft.ifft(pink).real
    elif noise_type == 'brown':
        noise = np.cumsum(np.random.randn(len(audio)))
        noise = noise / np.max(np.abs(noise))

    return audio + noise_factor * noise

def dynamic_range_compression(audio):
    """
    Use librosa.effects.percussive to isolate percussive components,
    which can act like a simple dynamic range compression approach.
    """
    return librosa.effects.percussive(audio)

def add_reverb(audio):
    """
    A simple 'reverb-like' effect using librosa's preemphasis as a stand-in.
    For a real reverb, you'd convolve with an IR, but this is a placeholder.
    """
    return librosa.effects.preemphasis(audio)

def equalize(audio):
    """
    Random amplitude scaling (0.8x to 1.2x) to mimic a simple EQ or loudness shift.
    """
    scale = np.random.uniform(0.8, 1.2)
    return audio * scale
