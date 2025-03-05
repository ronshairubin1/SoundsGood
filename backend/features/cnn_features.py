#!/usr/bin/env python3
"""
CNN Feature Preparation

This module handles the preparation of CNN-specific features from the unified feature set.
It extracts and formats the mel-spectrogram features in the shape required by CNN models.
"""

import numpy as np
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_extraction.log"),
        logging.StreamHandler()
    ]
)

class CNNFeaturePreparation:
    """
    Prepares CNN-specific features from the unified feature set.
    """
    
    def __init__(self, target_shape=(223, 64, 1)):
        """
        Initialize the CNN feature preparation.
        
        Args:
            target_shape: Target shape for CNN features (time, frequency, channels)
        """
        self.target_shape = target_shape
        
    def standardize_features(self, features):
        """
        Standardize features to the target shape.
        
        Args:
            features: Mel spectrogram features array
            
        Returns:
            Standardized features ready for CNN input
        """
        if features is None:
            return None
            
        try:
            # Get the current shape
            current_shape = features.shape
            
            # Check if we need to reshape
            if current_shape == self.target_shape:
                return features
                
            # Resize the time dimension (pad or trim)
            if current_shape[0] < self.target_shape[0]:
                # Pad
                padding = ((0, self.target_shape[0] - current_shape[0]), (0, 0))
                features = np.pad(features, padding, mode='constant')
            elif current_shape[0] > self.target_shape[0]:
                # Trim
                features = features[:self.target_shape[0], :]
                
            # Resize the frequency dimension (pad or trim)
            if current_shape[1] < self.target_shape[1]:
                # Pad
                padding = ((0, 0), (0, self.target_shape[1] - current_shape[1]))
                features = np.pad(features, padding, mode='constant')
            elif current_shape[1] > self.target_shape[1]:
                # Trim
                features = features[:, :self.target_shape[1]]
                
            # Add channel dimension if needed
            if len(features.shape) == 2:
                features = features[..., np.newaxis]
                
            return features
            
        except Exception as e:
            logging.error(f"Error standardizing CNN features: {str(e)}")
            return None
            
    def normalize_features(self, features):
        """
        Normalize feature values to [0, 1] range.
        
        Args:
            features: Feature array
            
        Returns:
            Normalized features
        """
        if features is None:
            return None
            
        try:
            # Get min and max values
            min_val = np.min(features)
            max_val = np.max(features)
            
            # Normalize to [0, 1]
            normalized = (features - min_val) / (max_val - min_val + 1e-10)
            
            return normalized
            
        except Exception as e:
            logging.error(f"Error normalizing CNN features: {str(e)}")
            return features
            
    def prepare_features(self, unified_features):
        """
        Prepare CNN-specific features from unified features.
        
        Args:
            unified_features: Unified feature dictionary
            
        Returns:
            Features ready for CNN input
        """
        if unified_features is None or 'mel_spectrogram' not in unified_features:
            logging.error("Missing mel_spectrogram in unified features")
            return None
            
        try:
            # Extract mel spectrogram
            mel_spec = unified_features['mel_spectrogram']
            
            # Standardize to target shape
            std_features = self.standardize_features(mel_spec)
            
            # Normalize values
            norm_features = self.normalize_features(std_features)
            
            return norm_features
            
        except Exception as e:
            logging.error(f"Error preparing CNN features: {str(e)}")
            return None
            
    def prepare_batch(self, unified_features_list):
        """
        Prepare a batch of CNN features from a list of unified features.
        
        Args:
            unified_features_list: List of unified feature dictionaries
            
        Returns:
            Numpy array of prepared features ready for CNN training
        """
        prepared_features = []
        
        for features in unified_features_list:
            prepared = self.prepare_features(features)
            if prepared is not None:
                prepared_features.append(prepared)
                
        if not prepared_features:
            return None
            
        return np.array(prepared_features) 