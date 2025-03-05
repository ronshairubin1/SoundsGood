import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import json
from datetime import datetime

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the unified feature extractor
from backend.features.extractor import FeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    # Get command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess all data for training')
    parser.add_argument('--data_dir', type=str, default='data/sounds', help='Directory containing sound classes')
    parser.add_argument('--features_dir', type=str, default='unified_features', help='Directory to save features')
    parser.add_argument('--cache_dir', type=str, default='.cache', help='Directory for caching features')
    args = parser.parse_args()
    
    data_dir = args.data_dir
    features_dir = args.features_dir
    cache_dir = args.cache_dir
    
    # Create directories if they don't exist
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Find all class directories
    class_dirs = [d for d in os.listdir(data_dir) 
                 if os.path.isdir(os.path.join(data_dir, d))]
    
    logging.info(f"Found {len(class_dirs)} class directories: {class_dirs}")
    
    # Initialize feature extractor
    logging.info("Initializing feature extractor...")
    extractor = FeatureExtractor(
        sample_rate=8000,
        n_mels=64,
        n_fft=1024,
        hop_length=256
    )
    
    # Extract CNN features
    logging.info("Extracting CNN features...")
    cnn_start = time.time()
    cnn_X, cnn_y, class_list, cnn_stats = extractor.batch_extract_features(
        data_dir, 
        class_dirs, 
        model_type='cnn',
        progress_callback=lambda percentage, message: logging.info(f"CNN Progress: {percentage}% - {message}")
    )
    cnn_time = time.time() - cnn_start
    
    # Extract RF features
    logging.info("Extracting RF features...")
    rf_start = time.time()
    rf_X, rf_y, _, rf_stats = extractor.batch_extract_features(
        data_dir, 
        class_dirs, 
        model_type='rf',
        progress_callback=lambda percentage, message: logging.info(f"RF Progress: {percentage}% - {message}")
    )
    rf_time = time.time() - rf_start
    
    # Print summary
    logging.info(f"CNN features shape: {cnn_X.shape}")
    logging.info(f"RF features shape: {rf_X.shape}")
    logging.info(f"CNN extraction time: {cnn_time:.2f} seconds")
    logging.info(f"RF extraction time: {rf_time:.2f} seconds")
    
    # Save feature information
    feature_info = {
        'cnn_shape': cnn_X.shape,
        'rf_shape': rf_X.shape,
        'class_list': class_list,
        'class_count': len(class_list),
        'cnn_extraction_time': cnn_time,
        'rf_extraction_time': rf_time,
        'timestamp': datetime.now().isoformat()
    }
    
    # Save the feature info
    with open(os.path.join(features_dir, 'feature_info.json'), 'w') as f:
        json.dump(feature_info, f, indent=4)
    
    # Print per-class statistics
    logging.info("Per-class statistics:")
    for class_dir in class_dirs:
        original = cnn_stats['original_counts'].get(class_dir, 0)
        cnn_processed = cnn_stats['processed_counts'].get(class_dir, 0)
        rf_processed = rf_stats['processed_counts'].get(class_dir, 0)
        logging.info(f"  {class_dir}: {original} files, {cnn_processed} CNN features, {rf_processed} RF features")
    
    # Save stats to file if features_dir is provided
    if features_dir:
        stats = {
            'cnn': cnn_stats,
            'rf': rf_stats,
            'class_list': class_list,
            'cnn_time': cnn_time,
            'rf_time': rf_time,
            'total_time': cnn_time + rf_time,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(os.path.join(features_dir, "preprocessing_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        
        # Save a simple plot of class distribution
        plt.figure(figsize=(12, 6))
        class_counts = np.bincount(np.array([class_list.index(c) for c in cnn_y]))
        plt.bar(class_list, class_counts)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(features_dir, "class_distribution.png"))

if __name__ == "__main__":
    main() 