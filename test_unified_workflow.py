#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to validate the unified feature extraction and model training workflow.

This script tests:
1. Feature extraction with the unified FeatureExtractor
2. Model training with the unified approach
3. Inference with the trained model

Usage:
    python test_unified_workflow.py
"""

import os
import sys
import logging
import time
import json
import numpy as np
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import the unified feature extractor
from backend.features.extractor import FeatureExtractor

# Import training service
from src.services.training_service import TrainingService

# Import for inference
from src.ml.inference import predict_sound

# Import config
from config import Config

def test_feature_extraction(audio_file):
    """Test feature extraction with the unified FeatureExtractor."""
    logging.info(f"Testing feature extraction with file: {audio_file}")
    
    if not os.path.exists(audio_file):
        logging.error(f"Audio file not found: {audio_file}")
        return False
    
    try:
        # Create feature extractor
        extractor = FeatureExtractor(sample_rate=16000)
        
        # Extract features
        start_time = time.time()
        all_features = extractor.extract_features(audio_file)
        extraction_time = time.time() - start_time
        
        if all_features is None:
            logging.error("Feature extraction failed")
            return False
        
        # Extract model-specific features
        cnn_features = extractor.extract_features_for_model(all_features, model_type='cnn')
        rf_features = extractor.extract_features_for_model(all_features, model_type='rf')
        
        # Log results
        logging.info(f"Feature extraction successful in {extraction_time:.2f} seconds")
        logging.info(f"CNN features shape: {cnn_features.shape}")
        logging.info(f"RF features count: {len(rf_features)}")
        
        return True
    except Exception as e:
        logging.error(f"Error in feature extraction: {str(e)}")
        return False

def test_model_training(model_type, dict_name):
    """Test model training with the unified approach."""
    logging.info(f"Testing {model_type} model training with dictionary: {dict_name}")
    
    try:
        # Create training service
        training_service = TrainingService()
        
        # Get dictionary data
        dictionaries = Config.get_dictionaries()
        selected_dict = None
        for d in dictionaries:
            if d.get('name') == dict_name:
                selected_dict = d
                break
        
        if not selected_dict:
            logging.error(f"Dictionary not found: {dict_name}")
            return False
        
        sounds = selected_dict.get('sounds', [])
        if not sounds:
            logging.error(f"No sounds found in dictionary: {dict_name}")
            return False
        
        # Get sounds directory
        sounds_dir = Config.TRAINING_SOUNDS_DIR
        if not os.path.exists(sounds_dir):
            logging.error(f"Sounds directory not found: {sounds_dir}")
            return False
        
        # Set up training parameters
        training_params = {
            'model_type': model_type,
            'audio_dir': sounds_dir,
            'save': True,
            'model_name': f"test_{dict_name}_{model_type}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'class_names': sounds,
            'dict_name': dict_name
        }
        
        # Add model-specific parameters
        if model_type == 'cnn':
            training_params.update({
                'epochs': 10,  # Reduced for testing
                'batch_size': 32,
                'use_class_weights': True,
                'use_data_augmentation': True
            })
        elif model_type == 'rf':
            training_params.update({
                'n_estimators': 50,  # Reduced for testing
                'max_depth': None
            })
        
        # Define progress callback
        def progress_callback(progress, status_message):
            logging.info(f"Training progress: {progress}%, Status: {status_message}")
        
        training_params['progress_callback'] = progress_callback
        
        # Train model
        start_time = time.time()
        result = training_service.train_unified(**training_params)
        training_time = time.time() - start_time
        
        # Check result
        if result and result.get('status') == 'success':
            logging.info(f"Training successful in {training_time:.2f} seconds")
            logging.info(f"Result: {result}")
            return True
        else:
            error_msg = result.get('message') if result and 'message' in result else "Unknown error"
            logging.error(f"Training failed: {error_msg}")
            return False
    except Exception as e:
        logging.error(f"Error in model training: {str(e)}")
        return False

def test_inference(model_type, dict_name, audio_file):
    """Test inference with the trained model."""
    logging.info(f"Testing inference with {model_type} model and file: {audio_file}")
    
    try:
        # Get model path
        if model_type == 'cnn':
            model_path = os.path.join('models', f"{dict_name.replace(' ', '_')}_model.h5")
        elif model_type == 'rf':
            model_path = os.path.join('models', f"{dict_name.replace(' ', '_')}_rf.joblib")
        else:
            logging.error(f"Unsupported model type for inference test: {model_type}")
            return False
        
        if not os.path.exists(model_path):
            logging.error(f"Model file not found: {model_path}")
            return False
        
        # Get class names
        dictionaries = Config.get_dictionaries()
        selected_dict = None
        for d in dictionaries:
            if d.get('name') == dict_name:
                selected_dict = d
                break
        
        if not selected_dict:
            logging.error(f"Dictionary not found: {dict_name}")
            return False
        
        sounds = selected_dict.get('sounds', [])
        
        # Load model
        if model_type == 'cnn':
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
            
            # Perform inference
            pred_class, confidence = predict_sound(model, audio_file, sounds)
            
            if pred_class:
                logging.info(f"Prediction: {pred_class} with confidence: {confidence:.2f}")
                return True
            else:
                logging.error("Prediction failed")
                return False
        else:
            logging.info("RF inference test not implemented yet")
            return True
    except Exception as e:
        logging.error(f"Error in inference test: {str(e)}")
        return False

def main():
    """Run all tests."""
    # Find a sample audio file
    sample_audio = None
    sounds_dir = Config.TRAINING_SOUNDS_DIR
    if os.path.exists(sounds_dir):
        for root, dirs, files in os.walk(sounds_dir):
            for file in files:
                if file.endswith('.wav'):
                    sample_audio = os.path.join(root, file)
                    break
            if sample_audio:
                break
    
    if not sample_audio:
        logging.error("No sample audio file found for testing")
        return False
    
    # Get a dictionary for testing
    dictionaries = Config.get_dictionaries()
    if not dictionaries:
        logging.error("No dictionaries found for testing")
        return False
    
    test_dict = dictionaries[0]['name']
    
    # Run tests
    logging.info("Starting unified workflow tests")
    
    # Test feature extraction
    logging.info("=== Testing Feature Extraction ===")
    feature_result = test_feature_extraction(sample_audio)
    
    # Test model training (CNN)
    logging.info("=== Testing CNN Model Training ===")
    cnn_training_result = test_model_training('cnn', test_dict)
    
    # Test model training (RF)
    logging.info("=== Testing RF Model Training ===")
    rf_training_result = test_model_training('rf', test_dict)
    
    # Test inference
    logging.info("=== Testing Inference ===")
    inference_result = test_inference('cnn', test_dict, sample_audio)
    
    # Report results
    logging.info("=== Test Results ===")
    logging.info(f"Feature Extraction: {'PASS' if feature_result else 'FAIL'}")
    logging.info(f"CNN Model Training: {'PASS' if cnn_training_result else 'FAIL'}")
    logging.info(f"RF Model Training: {'PASS' if rf_training_result else 'FAIL'}")
    logging.info(f"Inference: {'PASS' if inference_result else 'FAIL'}")
    
    return feature_result and cnn_training_result and rf_training_result and inference_result

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 