#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Integration tests for the unified audio processing pipeline.

This test script verifies that the new audio processing components work
correctly with the existing training and inference pipelines.
"""

import os
import sys
import unittest
import logging
import numpy as np
import tempfile
import librosa
import json
import time
import shutil
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the components we need to test
from backend.audio import AudioPreprocessor, AudioAugmentor, AudioRecorder
from backend.features.extractor import FeatureExtractor
from src.services.recording_service import RecordingService

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tests/output/audio_pipeline_test.log'),
        logging.StreamHandler()
    ]
)

class TestAudioPipeline(unittest.TestCase):
    """Test case for integrated audio processing pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directories for testing
        self.test_dir = tempfile.mkdtemp()
        self.training_dir = os.path.join(self.test_dir, 'training_sounds')
        self.augmented_dir = os.path.join(self.test_dir, 'augmented_sounds')
        
        os.makedirs(self.training_dir, exist_ok=True)
        os.makedirs(self.augmented_dir, exist_ok=True)
        
        # Generate a test audio file
        self.test_audio_path = os.path.join(self.test_dir, 'test_audio.wav')
        self._create_test_audio_file(self.test_audio_path)
        
        # Create a recording service for testing
        self.recording_service = RecordingService(
            data_dir=self.test_dir,
            training_dir=self.training_dir,
            augmented_dir=self.augmented_dir,
            temp_dir=self.test_dir
        )
        
        # Create direct component instances
        self.preprocessor = AudioPreprocessor()
        self.augmentor = AudioAugmentor(preprocessor=self.preprocessor)
        self.feature_extractor = FeatureExtractor()
        
        logging.info(f"Test environment set up in {self.test_dir}")
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir)
        logging.info("Test environment cleaned up")
    
    def _create_test_audio_file(self, file_path):
        """Create a test audio file with a sine wave."""
        # Generate a 1-second sine wave
        sample_rate = 16000
        t = np.linspace(0, 1, sample_rate)
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        
        # Add some silence before and after
        silence = np.zeros(int(sample_rate * 0.2))
        audio = np.concatenate([silence, audio, silence])
        
        # Save to WAV file
        import soundfile as sf
        sf.write(file_path, audio, sample_rate)
        
        logging.info(f"Created test audio file: {file_path}")
        return file_path
    
    def _create_multi_sound_test_file(self, file_path):
        """Create a test audio file with multiple sounds separated by silence."""
        # Generate several short sound segments
        sample_rate = 16000
        t = np.linspace(0, 0.3, int(sample_rate * 0.3))
        
        # Create 3 different tones
        tone1 = np.sin(2 * np.pi * 440 * t)  # 440 Hz
        tone2 = np.sin(2 * np.pi * 587 * t)  # 587 Hz
        tone3 = np.sin(2 * np.pi * 659 * t)  # 659 Hz
        
        # Create silence
        silence = np.zeros(int(sample_rate * 0.2))
        
        # Combine into a single audio stream with silences in between
        audio = np.concatenate([
            silence, tone1, silence,
            silence, tone2, silence,
            silence, tone3, silence
        ])
        
        # Save to WAV file
        import soundfile as sf
        sf.write(file_path, audio, sample_rate)
        
        logging.info(f"Created multi-sound test file: {file_path}")
        return file_path
    
    def test_audio_preprocessing(self):
        """Test basic audio preprocessing functionality."""
        logging.info("Testing audio preprocessing...")
        
        # Load test audio
        audio_data, sr = librosa.load(self.test_audio_path, sr=16000)
        
        # Test preprocessing
        processed_audio = self.preprocessor.preprocess_audio(audio_data)
        
        # Verify results
        self.assertIsNotNone(processed_audio)
        self.assertEqual(len(processed_audio), self.preprocessor.sample_rate * self.preprocessor.target_duration)
        
        logging.info("Audio preprocessing test passed")
    
    def test_audio_chopping(self):
        """Test audio chopping functionality."""
        logging.info("Testing audio chopping...")
        
        # Create multi-sound test file
        multi_sound_path = os.path.join(self.test_dir, 'multi_sound.wav')
        self._create_multi_sound_test_file(multi_sound_path)
        
        # Load test audio
        audio_data, sr = librosa.load(multi_sound_path, sr=16000)
        
        # Test chopping
        segments = self.preprocessor.chop_audio(audio_data)
        
        # Verify results
        self.assertIsNotNone(segments)
        self.assertGreater(len(segments), 1)  # Should find multiple segments
        
        # Test preprocessing each segment
        processed_segments = []
        for segment in segments:
            processed = self.preprocessor.preprocess_audio(segment)
            if processed is not None:
                processed_segments.append(processed)
        
        self.assertGreater(len(processed_segments), 0)
        
        logging.info(f"Audio chopping test passed: found {len(segments)} segments, "
                    f"processed {len(processed_segments)} successfully")
    
    def test_audio_augmentation(self):
        """Test audio augmentation functionality."""
        logging.info("Testing audio augmentation...")
        
        # Load test audio
        audio_data, sr = librosa.load(self.test_audio_path, sr=16000)
        
        # Preprocess the audio
        processed_audio = self.preprocessor.preprocess_audio(audio_data)
        
        # Test augmentation with multiple methods
        augmented_samples = self.augmentor.augment_audio_multiple(
            processed_audio, 
            methods=['pitch_shift', 'time_stretch', 'add_noise'],
            count=3
        )
        
        # Verify results
        self.assertIsNotNone(augmented_samples)
        self.assertEqual(len(augmented_samples), 3)
        
        logging.info(f"Audio augmentation test passed: created {len(augmented_samples)} augmented samples")
    
    def test_feature_extraction(self):
        """Test feature extraction with preprocessed audio."""
        logging.info("Testing feature extraction with preprocessed audio...")
        
        # Load and preprocess test audio
        audio_data, sr = librosa.load(self.test_audio_path, sr=16000)
        processed_audio = self.preprocessor.preprocess_audio(audio_data)
        
        # Extract features
        all_features = self.feature_extractor.extract_features(processed_audio, is_file=False)
        
        # Test feature extraction for different model types
        cnn_features = self.feature_extractor.extract_features_for_model(all_features, model_type='cnn')
        rf_features = self.feature_extractor.extract_features_for_model(all_features, model_type='rf')
        
        # Verify results
        self.assertIsNotNone(all_features)
        self.assertIsNotNone(cnn_features)
        self.assertIsNotNone(rf_features)
        
        # Verify CNN features shape
        self.assertEqual(len(cnn_features.shape), 2)  # Should be a 2D array
        
        # Verify RF features
        self.assertTrue(isinstance(rf_features, dict))
        self.assertGreater(len(rf_features), 0)
        
        logging.info("Feature extraction test passed")
    
    def test_save_training_sound(self):
        """Test saving preprocessed audio for training."""
        logging.info("Testing saving training sounds...")
        
        # Create a test class directory
        test_class = 'test_class'
        class_dir = os.path.join(self.training_dir, test_class)
        os.makedirs(class_dir, exist_ok=True)
        
        # Load and preprocess test audio
        audio_data, sr = librosa.load(self.test_audio_path, sr=16000)
        processed_audio = self.preprocessor.preprocess_audio(audio_data)
        
        # Save the preprocessed audio
        saved_path = self.preprocessor.save_training_sound(
            processed_audio,
            test_class,
            self.training_dir,
            metadata={'test': True}
        )
        
        # Verify the file was saved
        self.assertIsNotNone(saved_path)
        self.assertTrue(os.path.exists(saved_path))
        
        # Verify metadata was saved
        metadata_path = os.path.join(self.test_dir, 'sounds.json')
        self.assertTrue(os.path.exists(metadata_path))
        
        # Load and verify metadata
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.assertGreater(len(metadata), 0)
        
        # Verify the saved audio can be loaded and processed again
        saved_audio, saved_sr = librosa.load(saved_path, sr=16000)
        self.assertEqual(saved_sr, 16000)
        self.assertEqual(len(saved_audio), self.preprocessor.sample_rate * self.preprocessor.target_duration)
        
        logging.info(f"Save training sound test passed: saved to {saved_path}")
    
    def test_recording_service_integration(self):
        """Test the RecordingService integration with the audio processing components."""
        logging.info("Testing RecordingService integration...")
        
        # Create a test class
        test_class = 'test_service'
        
        # Test saving a sound through the service
        audio_data, sr = librosa.load(self.test_audio_path, sr=16000)
        
        # Save through the service
        output_path = self.recording_service.save_training_sound(
            audio_data,
            test_class,
            is_approved=True,
            metadata={'test': True}
        )
        
        # Verify the file was saved
        self.assertIsNotNone(output_path)
        self.assertTrue(os.path.exists(output_path))
        
        # Check the class directory was created
        class_dir = os.path.join(self.training_dir, test_class)
        self.assertTrue(os.path.exists(class_dir))
        
        # Verify the RecordingService can list classes
        classes = self.recording_service.get_available_classes()
        self.assertIn(test_class, classes)
        
        # Verify sample count
        count = self.recording_service.get_class_sample_count(test_class)
        self.assertEqual(count, 1)
        
        logging.info("RecordingService integration test passed")
    
    def test_end_to_end_workflow(self):
        """Test the complete end-to-end workflow from preprocessing to feature extraction."""
        logging.info("Testing end-to-end workflow...")
        
        # Create multi-sound test file
        multi_sound_path = os.path.join(self.test_dir, 'e2e_multi_sound.wav')
        self._create_multi_sound_test_file(multi_sound_path)
        
        # 1. Load test audio
        audio_data, sr = librosa.load(multi_sound_path, sr=16000)
        
        # 2. Chop audio into segments
        segments = self.preprocessor.chop_audio(audio_data)
        self.assertGreater(len(segments), 0)
        
        # 3. Process each segment
        processed_segments = []
        for segment in segments:
            processed = self.preprocessor.preprocess_audio(segment)
            if processed is not None:
                processed_segments.append(processed)
        
        self.assertGreater(len(processed_segments), 0)
        
        # 4. Save segments as training data
        test_class = 'e2e_test'
        saved_paths = []
        
        for i, segment in enumerate(processed_segments):
            saved_path = self.preprocessor.save_training_sound(
                segment,
                test_class,
                self.training_dir,
                metadata={'test': True, 'segment': i}
            )
            saved_paths.append(saved_path)
        
        self.assertGreater(len(saved_paths), 0)
        
        # 5. Create augmented versions
        augmented_paths = []
        for path in saved_paths:
            aug_class_dir = os.path.join(self.augmented_dir, test_class)
            os.makedirs(aug_class_dir, exist_ok=True)
            
            aug_paths = self.augmentor.augment_file(path, aug_class_dir, count=2)
            augmented_paths.extend(aug_paths)
        
        self.assertGreater(len(augmented_paths), 0)
        
        # 6. Extract features from all files
        original_features = []
        for path in saved_paths:
            all_features = self.feature_extractor.extract_features(path)
            cnn_features = self.feature_extractor.extract_features_for_model(all_features, model_type='cnn')
            rf_features = self.feature_extractor.extract_features_for_model(all_features, model_type='rf')
            
            self.assertIsNotNone(cnn_features)
            self.assertIsNotNone(rf_features)
            
            original_features.append((cnn_features, rf_features))
        
        # 7. Extract features from augmented files
        augmented_features = []
        for path in augmented_paths:
            all_features = self.feature_extractor.extract_features(path)
            cnn_features = self.feature_extractor.extract_features_for_model(all_features, model_type='cnn')
            rf_features = self.feature_extractor.extract_features_for_model(all_features, model_type='rf')
            
            self.assertIsNotNone(cnn_features)
            self.assertIsNotNone(rf_features)
            
            augmented_features.append((cnn_features, rf_features))
        
        # Verify we have all the features we need for training
        self.assertEqual(len(original_features), len(saved_paths))
        self.assertEqual(len(augmented_features), len(augmented_paths))
        
        logging.info(f"End-to-end workflow test passed: {len(saved_paths)} original samples, "
                    f"{len(augmented_paths)} augmented samples")

if __name__ == '__main__':
    unittest.main() 