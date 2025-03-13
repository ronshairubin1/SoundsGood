# FeatureExtractor has been moved to backend/features/extractor.py
# This file provides compatibility classes for the new architecture

import warnings
import os
import sys
import logging
import numpy as np
import traceback
import pickle as pickle_module  # Rename to avoid any potential shadowing

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.warn(
    "AudioFeatureExtractor has been replaced by FeatureExtractor in backend.features.extractor. "
    "Please update your imports to use 'from backend.features.extractor import FeatureExtractor' directly.",
    DeprecationWarning,
    stacklevel=2
)

# Make sure the parent directory is in the path so we can import the backend module
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
project_root = os.path.dirname(src_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    logger.info(f"Added {project_root} to Python path for imports")

# Now we can import from the backend directory
try:
    from backend.features.extractor import FeatureExtractor
    logger.info("Successfully imported FeatureExtractor from backend.features.extractor")
except ImportError as e:
    logger.error(f"ERROR importing FeatureExtractor: {e}")
    # Fall back to a direct import with absolute path
    sys.path.insert(0, os.path.join(project_root, 'backend'))
    logger.info(f"Added {os.path.join(project_root, 'backend')} to Python path")
    try:
        from features.extractor import FeatureExtractor
        logger.info("Successfully imported FeatureExtractor via direct import")
    except ImportError as e2:
        logger.error(f"CRITICAL: Still cannot import FeatureExtractor: {e2}")
        logger.error(f"Python path: {sys.path}")
        raise

# Add method to check if features exist for an audio file
original_feature_extractor = FeatureExtractor

# Extend the FeatureExtractor class to add the check_features_exist method
class FeatureExtractor(original_feature_extractor):
    def __init__(self, *args, **kwargs):
        # Call the parent class constructor
        super().__init__(*args, **kwargs)
        # Ensure we have a default cache_dir set
        if not hasattr(self, 'cache_dir'):
            # Set a default cache directory in the project root
            import os
            self.cache_dir = os.path.join(project_root, 'backend', 'cache', 'features')
            os.makedirs(self.cache_dir, exist_ok=True)
            logger.info(f"Using default cache directory: {self.cache_dir}")
            
    def check_features_exist(self, audio_file_path):
        """Check if features have already been extracted for a given audio file.
        
        Args:
            audio_file_path (str): Path to the audio file to check
            
        Returns:
            bool: True if features exist, False otherwise
        """
        try:
            # Generate a cache key based on the file path
            import hashlib
            file_hash = hashlib.md5(audio_file_path.encode()).hexdigest()
            
            # Check for feature files in the cache directory
            feature_types = ['mfcc', 'chroma', 'mel', 'contrast', 'tonnetz']
            for feature_type in feature_types:
                cache_file = os.path.join(self.cache_dir, f"{file_hash}_{feature_type}.npy")
                if not os.path.exists(cache_file):
                    # If any feature type is missing, return False
                    return False
            
            # All feature files exist
            return True
        except Exception as e:
            logger.error(f"Error checking features for {audio_file_path}: {e}")
            return False
            
    def get_cache_path(self, audio_file_path):
        """Get the path to the cached features for an audio file.
        
        Args:
            audio_file_path (str): Path to the audio file
            
        Returns:
            str: Path to the cached features directory or file
        """
        try:
            # Generate a cache key based on the file path
            import hashlib
            file_hash = hashlib.md5(audio_file_path.encode()).hexdigest()
            
            # Return the path to the mfcc features which is one of the main feature types
            # We use this as a representative cache file
            return os.path.join(self.cache_dir, f"{file_hash}_mfcc.npy")
        except Exception as e:
            logger.error(f"Error getting cache path for {audio_file_path}: {e}")
            # Return a path that will definitely not exist
            return os.path.join(self.cache_dir, "nonexistent_feature_file.npy")

# For backward compatibility, we alias the FeatureExtractor as AudioFeatureExtractor
# This allows old code to continue working while using the new implementation
AudioFeatureExtractor = FeatureExtractor

# Create a unified feature extractor that is used by the training service
class UnifiedFeatureExtractor:
    """Unified feature extractor that bridges between the training service and backend feature extractor."""
    
    def __init__(self, sample_rate=16000, n_mels=64, n_fft=1024, hop_length=256):
        """Initialize the unified feature extractor.
        
        Args:
            sample_rate: Audio sample rate
            n_mels: Number of mel bands for spectrogram (used for CNN model)
            n_fft: FFT window size
            hop_length: Hop length for STFT
        """
        logger.info(f"Initializing UnifiedFeatureExtractor with params: sample_rate={sample_rate}, n_mels={n_mels}, n_fft={n_fft}, hop_length={hop_length}")
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Create backend feature extractor with caching enabled
        try:
            cache_dir = os.path.join(project_root, 'backend', 'cache', 'features')
            os.makedirs(cache_dir, exist_ok=True)
            
            self.extractor = FeatureExtractor(
                sample_rate=sample_rate,
                n_mels=n_mels,
                n_fft=n_fft,
                hop_length=hop_length,
                cache_dir=cache_dir,
                use_cache=True
            )
            logger.info(f"Successfully created backend FeatureExtractor with cache_dir={cache_dir}")
        except Exception as e:
            logger.error(f"Error creating backend FeatureExtractor: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def extract_features(self, audio_source, is_file=True):
        """Extract all features from audio source.
        
        Args:
            audio_source: Path to audio file or audio data
            is_file: Whether audio_source is a file path or audio data
            
        Returns:
            Dictionary of features
        """
        try:
            logger.info(f"Extracting features with is_file={is_file}, audio_source type={type(audio_source)}")
            if not is_file and isinstance(audio_source, np.ndarray):
                audio_info = f"Audio array shape: {audio_source.shape}, dtype: {audio_source.dtype}"
                audio_info += f", min: {np.min(audio_source) if audio_source.size > 0 else 'empty'}"
                audio_info += f", max: {np.max(audio_source) if audio_source.size > 0 else 'empty'}"
                audio_info += f", has_nan: {np.isnan(audio_source).any()}"
                audio_info += f", duration: {len(audio_source)/self.sample_rate:.2f}s"
                logger.info(audio_info)
                
                # Warning for very short audio
                if len(audio_source)/self.sample_rate < 0.1:
                    logger.warning(f"Audio is very short ({len(audio_source)/self.sample_rate:.3f}s) - may not have enough data for feature extraction")
                    
                # Warning for audio with NaN values
                if np.isnan(audio_source).any():
                    logger.warning("Audio contains NaN values which will cause feature extraction to fail")
            elif is_file and isinstance(audio_source, str):
                logger.info(f"Processing audio file: {audio_source}")
                if not os.path.exists(audio_source):
                    logger.error(f"Audio file does not exist: {audio_source}")
                    return None
            
            # Call the backend extractor with enhanced error handling
            logger.info(f"Calling backend extractor with is_file={is_file}, audio type={type(audio_source)}")
            features = self.extractor.extract_features(audio_source, is_file=is_file)
            
            if features is not None:
                if isinstance(features, dict):
                    feature_info = f"Successfully extracted features with keys: {list(features.keys())}"
                    # Log additional details about each feature
                    for key, value in features.items():
                        if hasattr(value, 'shape'):
                            feature_info += f"\n - {key}: shape={value.shape}, dtype={value.dtype if hasattr(value, 'dtype') else 'N/A'}"
                            
                            # Check for NaN values in features
                            if hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.number):
                                nan_check = np.isnan(value).any() if hasattr(np.isnan(value), 'any') else False
                                feature_info += f", has_nan={nan_check}"
                                
                                if nan_check:
                                    logger.warning(f"Feature '{key}' contains NaN values which may cause issues")
                        else:
                            feature_info += f"\n - {key}: type={type(value)}"
                    logger.info(feature_info)
                else:
                    logger.info(f"Successfully extracted features of type: {type(features)}")
                    if hasattr(features, 'shape'):
                        logger.info(f"Feature shape: {features.shape}")
            else:
                logger.error("Feature extraction returned None - extraction failed")
            
            return features
        except Exception as e:
            logger.error(f"Error in extract_features: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(traceback.format_exc())
            # Log specific error types with additional context
            if isinstance(e, ValueError):
                logger.error("Possible causes: invalid audio format, empty audio, or NaN values")
            elif isinstance(e, (OSError, IOError)):
                logger.error("Possible causes: file not found, permission denied, or corrupted audio file")
            elif isinstance(e, MemoryError):
                logger.error("Possible causes: audio file too large or system low on memory")
            return None
    
    def extract_features_for_model(self, features, model_type='cnn'):
        """Extract model-specific features from raw features.
        
        Args:
            features: Raw features dictionary
            model_type: Type of model ('cnn', 'svm', etc.)
            
        Returns:
            Model-specific features
        """
        if features is None:
            logger.error("Cannot extract model-specific features from None")
            return None
        
        try:
            logger.info(f"Extracting {model_type}-specific features from {list(features.keys()) if isinstance(features, dict) else type(features)}")
            
            if model_type == 'cnn':
                # For CNN, we need mel spectrogram
                if isinstance(features, dict) and 'mel_spectrogram' in features:
                    mel_spec = features['mel_spectrogram']
                    if mel_spec is None:
                        logger.error("mel_spectrogram is None even though the key exists in features dictionary")
                        return None
                        
                    if hasattr(mel_spec, 'shape'):
                        logger.info(f"Using mel_spectrogram with shape {mel_spec.shape}, dtype={mel_spec.dtype if hasattr(mel_spec, 'dtype') else 'N/A'}")
                        
                        # Validate mel spectrogram dimensions and values
                        if len(mel_spec.shape) != 2:
                            logger.error(f"Invalid mel_spectrogram dimensions: expected 2D array, got {len(mel_spec.shape)}D")
                            return None
                            
                        if np.isnan(mel_spec).any():
                            logger.error("mel_spectrogram contains NaN values which will cause model training to fail")
                            return None
                            
                        # Apply proper normalization here rather than later
                        # Log before normalization stats
                        logger.info(f"Before normalization - mel_spec min: {mel_spec.min()}, max: {mel_spec.max()}, mean: {mel_spec.mean():.4f}")
                        
                        # Convert to dB scale (which is standard for spectrograms)
                        # Add small epsilon to avoid log(0)
                        eps = 1e-10
                        mel_spec_db = np.log10(mel_spec + eps)
                        
                        # Min-max normalization to [0,1] range
                        mel_spec_min = mel_spec_db.min()
                        mel_spec_max = mel_spec_db.max()
                        if mel_spec_max > mel_spec_min:  # Avoid division by zero
                            mel_spec = (mel_spec_db - mel_spec_min) / (mel_spec_max - mel_spec_min)
                        else:
                            logger.warning("Mel spectrogram has no dynamic range (min=max), using zeros")
                            mel_spec = np.zeros_like(mel_spec_db)
                        
                        # Log after normalization stats
                        logger.info(f"After normalization - mel_spec min: {mel_spec.min()}, max: {mel_spec.max()}, mean: {mel_spec.mean():.4f}")
                            
                        if mel_spec.shape[0] == 0 or mel_spec.shape[1] == 0:
                            logger.error(f"mel_spectrogram has zero dimension: {mel_spec.shape}")
                            return None
                            
                    else:
                        logger.warning(f"mel_spectrogram has no shape attribute, type is {type(mel_spec)}")
                        
                    # For CNN, discard the first MFCC coefficient as mentioned by the user
                    # This is a common practice since the first coefficient mainly represents the overall energy
                    if mel_spec.shape[0] > 1:  # Make sure we have enough coefficients
                        logger.info(f"Discarding first MFCC coefficient. Original shape: {mel_spec.shape}")
                        
                        # Skip the first coefficient - this is critical for model performance
                        mel_spec_without_first = mel_spec[1:, :]
                        
                        logger.info(f"After discarding first coefficient. New shape: {mel_spec_without_first.shape}")
                        
                        return mel_spec_without_first
                    else:
                        logger.warning("Not enough coefficients to discard the first one")
                        
                        return mel_spec
                else:
                    logger.error(f"mel_spectrogram not found in features. Available keys: {list(features.keys()) if isinstance(features, dict) else 'Not a dict'}")
                    return None
            elif model_type == 'svm':
                # For SVM, we need mfccs
                if isinstance(features, dict) and 'mfcc' in features:
                    mfcc = features['mfcc']
                    logger.info(f"Using mfcc with shape {mfcc.shape if hasattr(mfcc, 'shape') else 'unknown'}")
                    # For SVM we typically want the mean and std of MFCCs
                    if hasattr(mfcc, 'mean') and hasattr(mfcc, 'std'):
                        mfcc_features = np.hstack((mfcc.mean(axis=1), mfcc.std(axis=1)))
                        logger.info(f"Computed mfcc_features with shape {mfcc_features.shape if hasattr(mfcc_features, 'shape') else 'unknown'}")
                        return mfcc_features
                    else:
                        logger.error("mfcc does not have mean and std methods")
                        return None
                else:
                    logger.error(f"mfcc not found in features. Available keys: {list(features.keys()) if isinstance(features, dict) else 'Not a dict'}")
                    return None
            else:
                logger.error(f"Unsupported model type: {model_type}")
                return None
        except Exception as e:
            logger.error(f"Error in extract_features_for_model: {e}")
            logger.error(traceback.format_exc())
            return None
