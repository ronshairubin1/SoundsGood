# FeatureExtractor has been moved to backend/features/extractor.py
# This is a simple import forwarding file with no dependencies on legacy code

import warnings

warnings.warn(
    "AudioFeatureExtractor has been replaced by FeatureExtractor in backend.features.extractor. "
    "Please update your imports to use 'from backend.features.extractor import FeatureExtractor' directly.",
    DeprecationWarning,
    stacklevel=2
)

# Import only what's needed from the new location
from backend.features.extractor import FeatureExtractor

# For backward compatibility, we alias the FeatureExtractor as AudioFeatureExtractor
# This allows old code to continue working while using the new implementation
AudioFeatureExtractor = FeatureExtractor

# No other code or functionality should be in this file
