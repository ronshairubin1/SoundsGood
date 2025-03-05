"""
Unified audio processing module for training and inference.

This module provides a consistent approach to audio processing for both
training data preparation and inference, ensuring that the same processing
steps are applied in both cases.

Components:
- AudioPreprocessor: Handles audio preprocessing (chopping, finding boundaries, etc.)
- AudioAugmentor: Applies data augmentation to increase training dataset diversity
- AudioRecorder: Records and processes audio from microphone for training/inference
"""

from .preprocessor import AudioPreprocessor
from .augmentor import AudioAugmentor
from .recorder import AudioRecorder

__all__ = ['AudioPreprocessor', 'AudioAugmentor', 'AudioRecorder'] 