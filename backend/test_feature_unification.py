#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test Feature Unification

This script tests the unified feature extractor against legacy implementations
to ensure compatibility and correctness.

Usage:
    python test_feature_unification.py --audio_file path/to/audio.wav [--output_dir output_dir]
"""

import os
import sys
import logging
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import librosa
import tempfile

# Add the project root to the Python path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import the unified extractor
from backend.features.extractor import FeatureExtractor

# Import legacy extractors through forwarding mechanism (with deprecation warnings)
from src.ml.feature_extractor import AudioFeatureExtractor

# Import the unified extractor directly
from backend.features.extractor import FeatureExtractor as UnifiedExtractor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def extract_features_with_unified(audio_file):
    """Extract features using the unified extractor."""
    logging.info("Extracting features with unified extractor...")
    extractor = FeatureExtractor()
    features = extractor.extract_features(audio_file)
    return features

def extract_features_with_legacy(audio_file, legacy_type="audio"):
    """
    Extract features using a legacy extractor for comparison.
    
    Args:
        audio_file: Path to the audio file
        legacy_type: Type of legacy extractor to use
        
    Returns:
        Dict of extracted features
    """
    logging.info(f"Extracting features with legacy extractor ({legacy_type})...")
    
    features = None
    
    try:
        if legacy_type == "audio":
            # This now uses the forwarding mechanism to the unified extractor
            extractor = AudioFeatureExtractor()
            features = extractor.extract_features(audio_file)
        elif legacy_type == "unified":
            # This is just another reference to the unified extractor
            extractor = UnifiedExtractor()
            features = extractor.extract_features(audio_file)
        else:
            logging.warning(f"Unknown legacy type: {legacy_type}")
            return None
            
        return features
        
    except Exception as e:
        logging.error(f"Error extracting features with {legacy_type} extractor: {str(e)}")
        return None

def compare_feature_keys(unified_features, legacy_features, legacy_type):
    """
    Compare the keys of features extracted by different extractors
    
    Args:
        unified_features: Features extracted by the unified extractor
        legacy_features: Features extracted by a legacy extractor
        legacy_type: Type of legacy extractor used
        
    Returns:
        Dict with comparison results
    """
    if legacy_features is None:
        return {
            "legacy_type": legacy_type,
            "comparison_possible": False,
            "reason": "Legacy features not available"
        }
    
    # Flatten the unified features dictionary to get all keys
    unified_keys = set()
    
    def collect_keys(d, prefix=""):
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, (dict, list, tuple, np.ndarray)):
                    collect_keys(v, f"{prefix}{k}.")
                else:
                    unified_keys.add(f"{prefix}{k}")
    
    collect_keys(unified_features)
    
    # Get legacy keys
    legacy_keys = set()
    collect_keys(legacy_features)
    
    # Compare
    common_keys = unified_keys.intersection(legacy_keys)
    unified_only = unified_keys - legacy_keys
    legacy_only = legacy_keys - unified_keys
    
    return {
        "legacy_type": legacy_type,
        "comparison_possible": True,
        "common_key_count": len(common_keys),
        "unified_only_count": len(unified_only),
        "legacy_only_count": len(legacy_only),
        "common_keys": sorted(list(common_keys)),
        "unified_only_keys": sorted(list(unified_only)),
        "legacy_only_keys": sorted(list(legacy_only))
    }

def plot_feature_comparison(unified_features, all_legacy_features, output_file=None):
    """
    Plot a comparison of feature counts
    
    Args:
        unified_features: Features extracted by the unified extractor
        all_legacy_features: Dict of features extracted by legacy extractors
        output_file: Path to save the plot, or None to display
    """
    # Count features in the unified extractor
    unified_count = 0
    
    def count_features(d):
        count = 0
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, (int, float, bool, np.number)):
                    count += 1
                elif isinstance(v, (dict, list, tuple, np.ndarray)):
                    count += count_features(v)
        elif isinstance(d, (list, tuple, np.ndarray)):
            if len(d) > 0 and isinstance(d[0], (int, float, bool, np.number)):
                count += len(d)
            else:
                for item in d:
                    count += count_features(item)
        return count
    
    unified_count = count_features(unified_features)
    
    # Count features in legacy extractors
    legacy_counts = {}
    for legacy_type, features in all_legacy_features.items():
        if features is not None:
            legacy_counts[legacy_type] = count_features(features)
    
    # Plot
    plt.figure(figsize=(10, 6))
    
    # Create bar plot
    names = ["unified"] + list(legacy_counts.keys())
    values = [unified_count] + [legacy_counts[k] for k in legacy_counts.keys()]
    
    bars = plt.bar(names, values)
    
    # Add values on top of bars
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 5,
            str(value),
            ha='center',
            va='bottom'
        )
    
    plt.ylabel('Number of Features')
    plt.title('Feature Count Comparison Between Extractors')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        logging.info(f"Saved plot to {output_file}")
    else:
        plt.show()

def test_feature_extraction(audio_file, output_dir=None):
    """
    Test feature extraction with different extractors
    
    Args:
        audio_file: Path to the audio file
        output_dir: Directory to save outputs, or None for temporary directory
        
    Returns:
        Dict with test results
    """
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
        logging.info(f"Created temporary output directory: {output_dir}")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # Extract features with different extractors
    unified_features = extract_features_with_unified(audio_file)
    
    all_legacy_features = {
        "audio": extract_features_with_legacy(audio_file, "audio"),
        "unified": extract_features_with_legacy(audio_file, "unified"),
        "comprehensive": extract_features_with_legacy(audio_file, "comprehensive"),
        "advanced": extract_features_with_legacy(audio_file, "advanced")
    }
    
    # Compare features
    comparison_results = {}
    for legacy_type, legacy_features in all_legacy_features.items():
        if legacy_features is not None:
            comparison_results[legacy_type] = compare_feature_keys(unified_features, legacy_features, legacy_type)
    
    # Plot comparison
    plot_file = os.path.join(output_dir, "feature_comparison.png")
    plot_feature_comparison(unified_features, all_legacy_features, plot_file)
    
    # Save comparison results
    results_file = os.path.join(output_dir, "comparison_results.json")
    with open(results_file, "w") as f:
        json.dump(comparison_results, f, indent=2)
    
    logging.info(f"Saved comparison results to {results_file}")
    
    # Save extracted features
    unified_file = os.path.join(output_dir, "unified_features.npz")
    np.savez_compressed(unified_file, features=unified_features)
    
    for legacy_type, legacy_features in all_legacy_features.items():
        if legacy_features is not None:
            legacy_file = os.path.join(output_dir, f"{legacy_type}_features.npz")
            np.savez_compressed(legacy_file, features=legacy_features)
    
    logging.info(f"Saved extracted features to {output_dir}")
    
    return {
        "output_dir": output_dir,
        "comparison_results": comparison_results,
        "plot_file": plot_file,
        "results_file": results_file
    }

def main():
    parser = argparse.ArgumentParser(description='Test feature unification')
    parser.add_argument('--audio_file', required=True, help='Path to audio file for testing')
    parser.add_argument('--output_dir', help='Directory to save outputs')
    
    args = parser.parse_args()
    
    test_feature_extraction(args.audio_file, args.output_dir)

if __name__ == "__main__":
    main() 