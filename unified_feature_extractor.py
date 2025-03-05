# This file has been moved to the legacy folder.
# This import forwarder maintains backward compatibility.
# Please use the unified FeatureExtractor in backend/features/extractor.py for new code.

import warnings
import os

warnings.warn(
    "You are using a legacy feature extractor that has been moved to the legacy folder. "
    "Please use the unified FeatureExtractor in backend/features/extractor.py instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from the legacy location
legacy_path = os.path.join("legacy/feature_extractors", os.path.basename(__file__))
with open(legacy_path, "r") as f:
    exec(f.read())

# Also import the new extractor for convenience
try:
    from backend.features.extractor import FeatureExtractor
except ImportError:
    pass
