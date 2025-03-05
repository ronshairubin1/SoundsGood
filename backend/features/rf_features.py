#!/usr/bin/env python3
"""
Random Forest Feature Preparation

This module handles the preparation of Random Forest specific features from the unified feature set.
It extracts and formats the statistical features in the format required by RF models.
"""

import numpy as np
import logging
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_extraction.log"),
        logging.StreamHandler()
    ]
)

class RFFeaturePreparation:
    """
    Prepares Random Forest specific features from the unified feature set.
    """
    
    def __init__(self, exclude_first_mfcc=False):
        """
        Initialize the RF feature preparation.
        
        Args:
            exclude_first_mfcc: Whether to exclude the first MFCC coefficient
        """
        self.exclude_first_mfcc = exclude_first_mfcc
        
    def prepare_features(self, unified_features):
        """
        Prepare RF-specific features from unified features.
        
        Args:
            unified_features: Unified feature dictionary
            
        Returns:
            Dictionary of feature values ready for RF input
        """
        if unified_features is None or 'statistical' not in unified_features:
            logging.error("Missing statistical features in unified features")
            return None
            
        try:
            # Extract statistical features
            statistical = unified_features['statistical']
            
            # Create a copy to avoid modifying the original
            prepared_features = statistical.copy()
            
            # Optionally exclude the first MFCC coefficient
            if self.exclude_first_mfcc:
                keys_to_remove = [
                    'mfcc_0_mean', 'mfcc_0_std',
                    'mfcc_delta_0_mean', 'mfcc_delta_0_std',
                    'mfcc_delta2_0_mean', 'mfcc_delta2_0_std'
                ]
                for key in keys_to_remove:
                    if key in prepared_features:
                        del prepared_features[key]
            
            return prepared_features
            
        except Exception as e:
            logging.error(f"Error preparing RF features: {str(e)}")
            return None
    
    def get_feature_names(self, unified_features):
        """
        Get the names of features that will be used for RF model.
        
        Args:
            unified_features: Unified feature dictionary
            
        Returns:
            List of feature names
        """
        prepared = self.prepare_features(unified_features)
        if prepared is None:
            return []
        return list(prepared.keys())
    
    def prepare_batch(self, unified_features_list, labels=None):
        """
        Prepare a batch of RF features from a list of unified features.
        
        Args:
            unified_features_list: List of unified feature dictionaries
            labels: List of class labels (optional)
            
        Returns:
            DataFrame of prepared features ready for RF training
        """
        prepared_features = []
        
        for features in unified_features_list:
            prepared = self.prepare_features(features)
            if prepared is not None:
                prepared_features.append(prepared)
                
        if not prepared_features:
            return None
            
        # Convert to DataFrame
        df = pd.DataFrame(prepared_features)
        
        # Add labels if provided
        if labels is not None and len(labels) == len(prepared_features):
            df['label'] = labels
            
        return df
        
    def feature_vector_to_array(self, feature_dict):
        """
        Convert a feature dictionary to a numpy array in a consistent order.
        
        Args:
            feature_dict: Dictionary of feature values
            
        Returns:
            Numpy array of feature values
        """
        if feature_dict is None:
            return None
            
        # Get sorted keys for consistent order
        keys = sorted(feature_dict.keys())
        
        # Convert to array
        return np.array([feature_dict[key] for key in keys])
        
    def prepare_feature_arrays(self, unified_features_list, labels=None):
        """
        Prepare feature arrays from a list of unified features.
        
        Args:
            unified_features_list: List of unified feature dictionaries
            labels: List of class labels (optional)
            
        Returns:
            Tuple of (X, y) arrays for scikit-learn
        """
        prepared_features = []
        
        for features in unified_features_list:
            prepared = self.prepare_features(features)
            if prepared is not None:
                # Convert to array
                feature_array = self.feature_vector_to_array(prepared)
                prepared_features.append(feature_array)
                
        if not prepared_features:
            return None, None
            
        # Convert to numpy array
        X = np.array(prepared_features)
        
        # Return X and y
        if labels is not None and len(labels) == len(prepared_features):
            y = np.array(labels)
            return X, y
        else:
            return X, None 