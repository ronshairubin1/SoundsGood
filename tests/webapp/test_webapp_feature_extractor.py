#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to verify webapp integration with the unified FeatureExtractor.

This script tests:
1. The proper imports of FeatureExtractor in webapp files
2. The correct usage of feature extraction methods
3. Compatibility with the model inference pipeline
"""

import os
import sys
import logging
import unittest
import numpy as np
import json
from importlib import import_module

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tests/output/webapp_feature_test.log"),
        logging.StreamHandler()
    ]
)

class TestWebappFeatureExtractor(unittest.TestCase):
    """Test case for webapp integration with unified FeatureExtractor."""
    
    def setUp(self):
        """Set up test environment."""
        logging.info("Setting up test environment")
        self.test_audio_path = "backend/data/sounds/test_samples/test_sound.wav"
        
        # Create test audio file if it doesn't exist
        if not os.path.exists(self.test_audio_path):
            os.makedirs(os.path.dirname(self.test_audio_path), exist_ok=True)
            # Generate a simple test sound file
            try:
                import scipy.io.wavfile as wav
                import numpy as np
                
                # Generate a 1-second sine wave
                sample_rate = 16000
                t = np.linspace(0, 1, sample_rate)
                signal = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
                signal = (signal * 32767).astype(np.int16)
                
                # Save to a WAV file
                wav.write(self.test_audio_path, sample_rate, signal)
                logging.info(f"Created test audio file at {self.test_audio_path}")
            except Exception as e:
                logging.error(f"Failed to create test audio: {str(e)}")
                # Use a fallback path if we couldn't create the test file
                self.test_audio_path = self._find_audio_file()
    
    def _find_audio_file(self):
        """Find an existing audio file for testing if we can't create one."""
        logging.info("Looking for existing audio file...")
        for root, _, files in os.walk("data/sounds"):
            for file in files:
                if file.endswith(".wav"):
                    return os.path.join(root, file)
        return None
    
    def test_import_feature_extractor(self):
        """Test that the FeatureExtractor can be imported in the webapp."""
        logging.info("Testing FeatureExtractor import in webapp modules...")
        
        # Test sound detector modules
        try:
            from src.ml.sound_detector_rf import SoundDetectorRF
            detector = SoundDetectorRF()
            self.assertIsNotNone(detector.feature_extractor)
            logging.info("✅ SoundDetectorRF uses unified FeatureExtractor")
        except Exception as e:
            logging.error(f"❌ Error in SoundDetectorRF: {str(e)}")
            self.fail(f"SoundDetectorRF import failed: {str(e)}")
        
        try:
            from src.ml.sound_detector_ensemble import SoundDetectorEnsemble
            detector = SoundDetectorEnsemble()
            self.assertIsNotNone(detector.feature_extractor)
            logging.info("✅ SoundDetectorEnsemble uses unified FeatureExtractor")
        except Exception as e:
            logging.error(f"❌ Error in SoundDetectorEnsemble: {str(e)}")
            self.fail(f"SoundDetectorEnsemble import failed: {str(e)}")
        
        # Test inference module
        try:
            from src.ml.inference import SoundDetector
            detector = SoundDetector(model_type="cnn")
            self.assertIsNotNone(detector.feature_extractor)
            logging.info("✅ SoundDetector uses unified FeatureExtractor")
        except Exception as e:
            logging.error(f"❌ Error in SoundDetector: {str(e)}")
            self.fail(f"SoundDetector import failed: {str(e)}")
    
    def test_feature_extraction_in_webapp(self):
        """Test feature extraction functionality in webapp components."""
        logging.info("Testing feature extraction in webapp components...")
        
        if not self.test_audio_path or not os.path.exists(self.test_audio_path):
            self.skipTest("No test audio file available")
        
        # Test SoundDetectorRF feature extraction
        try:
            from src.ml.sound_detector_rf import SoundDetectorRF
            detector = SoundDetectorRF()
            
            # Load audio file
            import librosa
            audio, _ = librosa.load(self.test_audio_path, sr=16000)
            
            # Extract features using the detector
            prediction = detector.predict(audio)
            self.assertIsNotNone(prediction)
            logging.info("✅ SoundDetectorRF feature extraction works")
        except Exception as e:
            logging.error(f"❌ Error in SoundDetectorRF feature extraction: {str(e)}")
            self.fail(f"SoundDetectorRF feature extraction failed: {str(e)}")
        
        # Test SoundDetector feature extraction
        try:
            from src.ml.inference import SoundDetector
            detector = SoundDetector(model_type="cnn")
            
            # Test feature extraction without model prediction
            import librosa
            audio, _ = librosa.load(self.test_audio_path, sr=16000)
            all_features = detector.feature_extractor.extract_features(audio, is_file=False)
            cnn_features = detector.feature_extractor.extract_features_for_model(all_features, model_type='cnn')
            
            self.assertIsNotNone(all_features)
            self.assertIsNotNone(cnn_features)
            logging.info("✅ SoundDetector feature extraction works")
        except Exception as e:
            logging.error(f"❌ Error in SoundDetector feature extraction: {str(e)}")
            self.fail(f"SoundDetector feature extraction failed: {str(e)}")
    
    def test_cache_directory_structure(self):
        """Test that the cache directory structure is correct."""
        cache_dir = "backend/data/features/cache"
        self.assertTrue(os.path.exists(cache_dir), f"Cache directory {cache_dir} does not exist")
        logging.info(f"✅ Cache directory structure is correct: {cache_dir}")

def main():
    """Main function to run the tests."""
    unittest.main()

if __name__ == "__main__":
    main() 