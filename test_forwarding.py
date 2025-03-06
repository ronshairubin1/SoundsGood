#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script to verify the forwarding mechanism for legacy feature extractors.
"""

import os
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_legacy_imports():
    """Test that legacy imports still work through forwarding."""
    try:
        logging.info("Importing AudioFeatureExtractor (which should forward to FeatureExtractor)...")
        from src.ml.feature_extractor import AudioFeatureExtractor
        logging.info("Import successful")
        
        logging.info("Creating instance...")
        extractor = AudioFeatureExtractor()
        logging.info(f"Instance created: {extractor}")
        
        logging.info("Getting feature names...")
        feature_names = extractor.get_feature_names()
        logging.info(f"Feature names: {len(feature_names)} features available")
        
        # Verify this is actually the unified extractor
        logging.info("Verifying this is the unified extractor...")
        from backend.features.extractor import FeatureExtractor
        assert isinstance(extractor, FeatureExtractor), "Extractor is not an instance of FeatureExtractor"
        logging.info("Verification successful - forwarding is working correctly")
        
        return True
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return False

def test_unified_imports():
    """Test that unified imports work directly."""
    try:
        logging.info("Importing FeatureExtractor directly...")
        from backend.features.extractor import FeatureExtractor
        logging.info("Import successful")
        
        logging.info("Creating instance...")
        extractor = FeatureExtractor()
        logging.info(f"Instance created: {extractor}")
        
        logging.info("Getting feature names...")
        feature_names = extractor.get_feature_names()
        logging.info(f"Feature names: {len(feature_names)} features available")
        
        # Test feature extraction
        logging.info("Testing feature extraction methods...")
        methods = [method for method in dir(extractor) if method.startswith('extract_') and callable(getattr(extractor, method))]
        logging.info(f"Available extraction methods: {methods}")
        
        return True
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    logging.info("Testing legacy imports...")
    legacy_result = test_legacy_imports()
    logging.info(f"Legacy imports test {'passed' if legacy_result else 'failed'}")
    
    logging.info("\nTesting unified imports...")
    unified_result = test_unified_imports()
    logging.info(f"Unified imports test {'passed' if unified_result else 'failed'}")
    
    if legacy_result and unified_result:
        logging.info("\nAll tests passed!")
        sys.exit(0)
    else:
        logging.error("\nSome tests failed!")
        sys.exit(1) 