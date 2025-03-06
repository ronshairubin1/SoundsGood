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
    """Test that legacy imports still work."""
    try:
        logging.info("Importing AudioFeatureExtractor...")
        from src.ml.feature_extractor import AudioFeatureExtractor
        logging.info("Import successful")
        
        logging.info("Creating instance...")
        extractor = AudioFeatureExtractor()
        logging.info(f"Instance created: {extractor}")
        
        logging.info("Getting feature names...")
        feature_names = extractor.get_feature_names()
        logging.info(f"Feature names: {len(feature_names)} features available")
        
        return True
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return False

def test_unified_imports():
    """Test that unified imports work."""
    try:
        logging.info("Importing FeatureExtractor...")
        from backend.features.extractor import FeatureExtractor
        logging.info("Import successful")
        
        logging.info("Creating instance...")
        extractor = FeatureExtractor()
        logging.info(f"Instance created: {extractor}")
        
        return True
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing legacy imports...")
    legacy_result = test_legacy_imports()
    print(f"Legacy imports: {'✅ SUCCESS' if legacy_result else '❌ FAILED'}")
    
    print("\nTesting unified imports...")
    unified_result = test_unified_imports()
    print(f"Unified imports: {'✅ SUCCESS' if unified_result else '❌ FAILED'}")
    
    if legacy_result and unified_result:
        print("\n✅ All tests passed! The forwarding mechanism is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Tests failed. The forwarding mechanism is not working correctly.")
        sys.exit(1) 