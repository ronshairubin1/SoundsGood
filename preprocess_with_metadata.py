#!/usr/bin/env python3
import os
import sys
import logging
import time
import argparse
from tqdm import tqdm
import numpy as np
from datetime import datetime
import json
import glob

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the unified feature extractor
from backend.features.extractor import FeatureExtractor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocessing.log"),
        logging.StreamHandler()
    ]
)

def get_metadata_for_file(file_path):
    """
    Get metadata for an audio file.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        dict: Metadata for the file
    """
    # Get creation date, if possible
    try:
        ctime = os.path.getctime(file_path)
        ctime_str = datetime.fromtimestamp(ctime).isoformat()
    except:
        ctime_str = None
    
    # Get modification date, if possible
    try:
        mtime = os.path.getmtime(file_path)
        mtime_str = datetime.fromtimestamp(mtime).isoformat()
    except:
        mtime_str = None
    
    # Get file size
    try:
        size = os.path.getsize(file_path)
    except:
        size = None
    
    # Parse class from filename or path
    file_name = os.path.basename(file_path)
    parent_dir = os.path.basename(os.path.dirname(file_path))
    
    # Try to find a class identifier
    class_id = parent_dir
    
    # If filename starts with a known class, use that
    for prefix in ['eh', 'ah', 'ee', 'oo', 'oh']:
        if file_name.startswith(prefix + '_'):
            class_id = prefix
            break
    
    return {
        'file_path': file_path,
        'file_name': file_name,
        'creation_time': ctime_str,
        'modification_time': mtime_str,
        'size_bytes': size,
        'class': class_id,
        'processing_time': datetime.now().isoformat()
    }

def preprocess_files_with_metadata(input_glob, output_dir=None, metadata_file=None):
    """
    Preprocess files matching the input glob and save their features with metadata.
    
    Args:
        input_glob: Glob pattern for input files
        output_dir: Directory to save feature files
        metadata_file: File to save metadata
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize the extractor
    extractor = FeatureExtractor(
        sample_rate=8000,
        n_mels=64,
        n_fft=1024,
        hop_length=256
    )
    
    # Find all files matching the glob pattern
    files = glob.glob(input_glob)
    if not files:
        logging.error(f"No files found matching pattern: {input_glob}")
        return []
    
    logging.info(f"Found {len(files)} files matching pattern: {input_glob}")
    
    # Process each file
    all_metadata = []
    for file_path in files:
        try:
            # Get metadata
            metadata = get_metadata_for_file(file_path)
            
            # Extract features
            start_time = time.time()
            features = extractor.extract_features(file_path, is_file=True)
            processing_time = time.time() - start_time
            
            if features is None:
                logging.warning(f"Failed to extract features from {file_path}")
                continue
            
            # Add processing time to metadata
            metadata['feature_extraction_time'] = processing_time
            metadata['feature_extraction_timestamp'] = datetime.now().isoformat()
            metadata['feature_count'] = len(features)
            
            # Save features if output directory is provided
            if output_dir:
                file_name = os.path.basename(file_path)
                file_base = os.path.splitext(file_name)[0]
                feature_file = os.path.join(output_dir, f"{file_base}_features.npz")
                
                # Add feature metadata
                features['metadata'].update(metadata)
                
                # Save as npz
                np.savez_compressed(feature_file, features=features)
                logging.info(f"Saved features for {file_path} to {feature_file}")
            
            # Add to metadata list
            all_metadata.append(metadata)
            
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
    
    # Save metadata if metadata file is provided
    if metadata_file and all_metadata:
        with open(metadata_file, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        logging.info(f"Saved metadata to {metadata_file}")
    
    return all_metadata

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Preprocess audio files with metadata management")
    parser.add_argument('--data-dir', default='data', help='Base data directory')
    parser.add_argument('--sounds-dir', default='sounds/training_sounds', help='Sound files directory (relative to data-dir)')
    parser.add_argument('--cache-dir', default='feature_cache', help='Directory for feature cache')
    parser.add_argument('--saved-features-dir', default='saved_features', help='Directory for saved feature files')
    parser.add_argument('--scan-only', action='store_true', help='Only scan and update metadata without processing')
    parser.add_argument('--force-reprocess', action='store_true', help='Force reprocessing of all files')
    parser.add_argument('--include-excluded', action='store_true', help='Include files marked as excluded in processing')
    parser.add_argument('--input', required=True, help='Glob pattern for input files (e.g. "sounds/*.wav")')
    parser.add_argument('--output_dir', help='Directory to save feature files')
    parser.add_argument('--metadata_file', help='File to save metadata')
    args = parser.parse_args()
    
    # Initialize dataset manager
    logging.info("Initializing dataset manager...")
    manager = SoundDatasetManager(
        data_dir=args.data_dir,
        sounds_dir=args.sounds_dir,
        feature_dir=args.saved_features_dir
    )
    
    # Scan directory to update metadata
    logging.info("Scanning sound directories...")
    scan_results = manager.scan_directory()
    logging.info(f"Found {scan_results['total_files']} total files ({scan_results['new_files']} new)")
    
    # Show dataset summary
    summary = manager.get_dataset_summary()
    logging.info("Dataset summary:")
    logging.info(f"  Total files: {summary['total_files']}")
    logging.info(f"  Included in training: {summary['included_files']}")
    logging.info(f"  Excluded from training: {summary['excluded_files']}")
    logging.info("Classes:")
    for class_name, stats in summary['classes'].items():
        logging.info(f"  {class_name}: {stats['included']}/{stats['total']} files included")
    
    # Exit if scan-only mode
    if args.scan_only:
        logging.info("Scan-only mode, exiting without processing files")
        return
    
    # Create output directories
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, args.saved_features_dir), exist_ok=True)
    
    # Process files
    logging.info(f"Processing {len(files)} files...")
    processed = 0
    skipped = 0
    failed = 0
    cnn_shapes = []
    rf_feature_counts = []
    
    # Create class indices mapping (for labels)
    class_names = sorted(list(manager.metadata["classes"].keys()))
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    # Prepare arrays for storing features
    all_cnn_features = []
    all_cnn_labels = []
    all_rf_features = []
    all_rf_labels = []
    all_filenames = []  # Keep track of processed files
    
    # Process each file
    for file_path in tqdm(files, desc="Processing files"):
        try:
            if not os.path.exists(file_path):
                logging.error(f"File not found: {file_path}")
                failed += 1
                continue
            
            # Get class from file path
            rel_path = os.path.relpath(file_path, str(manager.sounds_dir))
            class_name = rel_path.split(os.sep)[0]
            
            # Check if already in cache (unless forcing reprocess)
            if not args.force_reprocess and extractor._check_cache(extractor._hash_file_path(file_path)):
                features = extractor.extract_features(file_path, is_file=True)
                skipped += 1
            else:
                # Extract features
                features = extractor.extract_features(file_path, is_file=True)
                
            if features is not None:
                processed += 1
                
                # Record shapes for checking consistency
                if 'cnn_features' in features:
                    cnn_shape = features['cnn_features'].shape
                    cnn_shapes.append(cnn_shape)
                    # Store features and label
                    all_cnn_features.append(features['cnn_features'])
                    all_cnn_labels.append(class_to_idx[class_name])
                
                if 'rf_features' in features:
                    rf_feature_count = len(features['rf_features'])
                    rf_feature_counts.append(rf_feature_count)
                    
                    # Convert RF features dictionary to vector
                    if not all_rf_features:
                        # Get feature names from first sample
                        rf_feature_names = sorted([k for k in features['rf_features'].keys() 
                                                 if k != 'first_mfcc_excluded'])
                    
                    # Create vector from dictionary
                    rf_vector = [features['rf_features'][k] for k in rf_feature_names]
                    all_rf_features.append(rf_vector)
                    all_rf_labels.append(class_to_idx[class_name])
                
                # Store filename
                all_filenames.append(rel_path)
            else:
                failed += 1
                
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            failed += 1
    
    # Convert lists to numpy arrays
    if all_cnn_features:
        logging.info("Preparing CNN features for saving...")
        X_cnn = np.array(all_cnn_features)
        y_cnn = np.array(all_cnn_labels)
        
        # Save CNN features
        cnn_save_path = os.path.join(args.data_dir, args.saved_features_dir, 'cnn_features.npz')
        np.savez_compressed(
            cnn_save_path,
            features=X_cnn,
            labels=y_cnn,
            filenames=all_filenames,
            class_names=class_names
        )
        logging.info(f"Saved CNN features to {cnn_save_path}")
        logging.info(f"CNN features shape: {X_cnn.shape}")
    
    if all_rf_features:
        logging.info("Preparing RF features for saving...")
        X_rf = np.array(all_rf_features)
        y_rf = np.array(all_rf_labels)
        
        # Save RF features
        rf_save_path = os.path.join(args.data_dir, args.saved_features_dir, 'rf_features.npz')
        np.savez_compressed(
            rf_save_path,
            features=X_rf,
            labels=y_rf,
            filenames=all_filenames,
            feature_names=rf_feature_names,
            class_names=class_names
        )
        logging.info(f"Saved RF features to {rf_save_path}")
        logging.info(f"RF features shape: {X_rf.shape}")
    
    # Save metadata about the features
    feature_info = {
        "timestamp": datetime.now().isoformat(),
        "num_samples": len(all_filenames),
        "class_to_idx": class_to_idx,
        "class_names": class_names,
        "cnn_shape": tuple(X_cnn.shape) if len(all_cnn_features) > 0 else None,
        "rf_shape": tuple(X_rf.shape) if len(all_rf_features) > 0 else None,
        "rf_feature_names": rf_feature_names if len(all_rf_features) > 0 else None,
        "processed_files": processed,
        "skipped_files": skipped,
        "failed_files": failed
    }
    
    with open(os.path.join(args.data_dir, args.saved_features_dir, 'feature_info.json'), 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    # Create a helper script for loading the features
    load_script_path = os.path.join(args.data_dir, args.saved_features_dir, 'load_features.py')
    with open(load_script_path, 'w') as f:
        f.write("""
import numpy as np

def load_cnn_features():
    \"\"\"Load CNN features from saved file.\"\"\"
    data = np.load('cnn_features.npz')
    X = data['features']
    y = data['labels']
    class_names = data['class_names']
    filenames = data['filenames']
    # Create class to index mapping
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    return X, y, class_names, class_to_idx, filenames

def load_rf_features():
    \"\"\"Load RF features from saved file.\"\"\"
    data = np.load('rf_features.npz')
    X = data['features']
    y = data['labels']
    feature_names = data['feature_names']
    class_names = data['class_names']
    filenames = data['filenames']
    # Create class to index mapping
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    return X, y, feature_names, class_names, class_to_idx, filenames
""")
    
    # Print summary
    logging.info("\n================ PREPROCESSING SUMMARY ================")
    logging.info(f"Total files processed: {processed + skipped}")
    logging.info(f"  Successfully processed: {processed}")
    logging.info(f"  Skipped (already cached): {skipped}")
    logging.info(f"  Failed: {failed}")
    
    # Print feature consistency info
    if cnn_shapes:
        unique_shapes = set(str(s) for s in cnn_shapes)
        logging.info(f"CNN shapes: {', '.join(unique_shapes)}")
    
    if rf_feature_counts:
        unique_counts = set(rf_feature_counts)
        logging.info(f"RF feature counts: {', '.join(str(c) for c in unique_counts)}")
    
    # Print class distribution
    class_counts = {name: 0 for name in class_names}
    for rel_path in all_filenames:
        class_name = rel_path.split(os.sep)[0]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    logging.info("\nClass distribution:")
    for class_name, count in class_counts.items():
        logging.info(f"  {class_name}: {count} samples")
    
    total_time = time.time() - start_time
    logging.info(f"\nTotal preprocessing time: {total_time:.2f} seconds")
    logging.info("================================================")

if __name__ == "__main__":
    main() 