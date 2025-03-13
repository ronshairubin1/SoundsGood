import os
import numpy as np
import librosa
import logging
import soundfile as sf
from scipy.io import wavfile
from scipy.signal import find_peaks
from scipy import signal
from scipy import stats
import pyloudnorm as pyln
import joblib  # For compatibility with model serialization
import warnings

# Import the bridge implementation
from backend.src.core.audio.processor_bridge import AudioProcessor

# Show a deprecation warning for direct imports from this file
warnings.warn(
    "Importing directly from src.core.audio.processor is deprecated. "
    "Please update your imports to use 'from backend.audio.processor import SoundProcessor'.",
    DeprecationWarning, stacklevel=2
)

# The original AudioProcessor class is now in processor_bridge.py
# This file just re-exports it for backward compatibility

# Remove all code below this line and just use the bridge