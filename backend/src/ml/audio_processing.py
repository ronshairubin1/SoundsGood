# SoundProcessor has been moved to backend/audio/processor.py
# This is a simple import forwarding file with no dependencies on legacy code

import warnings

warnings.warn(
    "SoundProcessor has been moved to backend.audio.processor. "
    "Please update your imports to use 'from backend.audio.processor import SoundProcessor' directly.",
    DeprecationWarning,
    stacklevel=2
)

# Import only what's needed from the new location
from backend.audio.processor import SoundProcessor

# No other code or functionality should be in this file
