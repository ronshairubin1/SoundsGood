#!/usr/bin/env python3
"""
Unified Dataset Manager

This module provides functionality for managing the dataset of sound files
and their extracted features. It handles file organization, metadata tracking,
and feature storage/retrieval.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_management.log"),
        logging.StreamHandler()
    ]
)

class DatasetManager:
    """
    Manages a dataset of sound files and their features for training and evaluation.
    Provides consistent organization and access to data across training and inference.
    """
    
    def __init__(self, base_dir="backend/data", 
                 sounds_dir="sounds", features_dir="features",
                 metadata_file="dataset_metadata.json"):
        """
        Initialize the dataset manager.
        
        Args:
            base_dir: Base directory for all data
            sounds_dir: Directory for sound files relative to base_dir
            features_dir: Directory for features relative to base_dir
            metadata_file: File to store dataset metadata
        """
        self.base_dir = Path(base_dir)
        
        # Set up sounds directory paths
        self.sounds_dir = self.base_dir / sounds_dir
        self.raw_sounds_dir = self.sounds_dir / "raw"
        self.training_sounds_dir = self.sounds_dir / "training_sounds"  # Renamed from 'chopped' to be more descriptive
        self.augmented_sounds_dir = self.sounds_dir / "augmented"
        
        # Set up feature directory paths
        self.features_dir = self.base_dir / features_dir
        self.unified_features_dir = self.features_dir / "unified"
        self.model_specific_features_dir = self.features_dir / "model_specific"
        
        # Metadata file path
        self.metadata_path = self.base_dir / metadata_file
        
        # Create directories if they don't exist
        self._create_directories()
        
        # Load or initialize metadata
        self.metadata = self._load_metadata()
        
    def _create_directories(self):
        """Create all necessary directories."""
        directories = [
            self.raw_sounds_dir,
            self.training_sounds_dir,
            self.augmented_sounds_dir,
            self.unified_features_dir,
            self.model_specific_features_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def _load_metadata(self):
        """Load metadata from file or initialize if it doesn't exist."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error loading metadata: {str(e)}")
                return self._initialize_metadata()
        else:
            return self._initialize_metadata()
            
    def _initialize_metadata(self):
        """Initialize a new metadata structure."""
        metadata = {
            'dataset_info': {
                'created_at': datetime.now().isoformat(),
                'last_updated': datetime.now().isoformat(),
                'version': '1.0'
            },
            'sound_classes': {},
            'files': {}
        }
        
        # Save the initialized metadata
        self._save_metadata(metadata)
        
        return metadata
        
    def _save_metadata(self, metadata=None):
        """Save metadata to file."""
        if metadata is None:
            metadata = self.metadata
            
        # Update last_updated timestamp
        metadata['dataset_info']['last_updated'] = datetime.now().isoformat()
        
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving metadata: {str(e)}")
            
    def add_class(self, class_name, description=None):
        """
        Add a new sound class to the dataset.
        
        Args:
            class_name: Name of the sound class
            description: Optional description of the class
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if class already exists
            if class_name in self.metadata['sound_classes']:
                logging.warning(f"Class '{class_name}' already exists")
                return False
                
            # Add class to metadata
            self.metadata['sound_classes'][class_name] = {
                'description': description or '',
                'created_at': datetime.now().isoformat(),
                'file_count': 0
            }
            
            # Create class directories
            for directory in [self.raw_sounds_dir, self.training_sounds_dir, self.augmented_sounds_dir]:
                class_dir = directory / class_name
                class_dir.mkdir(exist_ok=True)
                
            # Save metadata
            self._save_metadata()
            
            return True
            
        except Exception as e:
            logging.error(f"Error adding class '{class_name}': {str(e)}")
            return False
            
    def add_file(self, file_path, class_name, include_in_training=True):
        """
        Add a file to the dataset.
        
        Args:
            file_path: Path to the source file
            class_name: Name of the sound class
            include_in_training: Whether to include in training
            
        Returns:
            Path to the copied file if successful, None otherwise
        """
        try:
            # Check if class exists
            if class_name not in self.metadata['sound_classes']:
                logging.error(f"Class '{class_name}' does not exist")
                return None
                
            # Create a proper filename
            source_path = Path(file_path)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{class_name}_{timestamp}{source_path.suffix}"
            
            # Copy file to raw sounds directory
            dest_dir = self.raw_sounds_dir / class_name
            dest_path = dest_dir / filename
            
            # Copy the file
            shutil.copy2(file_path, dest_path)
            
            # Add to metadata
            file_id = str(dest_path.relative_to(self.base_dir))
            self.metadata['files'][file_id] = {
                'class': class_name,
                'added_at': datetime.now().isoformat(),
                'include_in_training': include_in_training,
                'original_path': str(source_path),
                'preprocessing_status': 'raw'
            }
            
            # Increment class file count
            self.metadata['sound_classes'][class_name]['file_count'] += 1
            
            # Save metadata
            self._save_metadata()
            
            return dest_path
            
        except Exception as e:
            logging.error(f"Error adding file '{file_path}': {str(e)}")
            return None
            
    def get_files_by_class(self, class_name, include_only_training=True):
        """
        Get all files for a specific class.
        
        Args:
            class_name: Name of the sound class
            include_only_training: Whether to include only training files
            
        Returns:
            List of file paths
        """
        files = []
        
        # Check if class exists
        if class_name not in self.metadata['sound_classes']:
            logging.error(f"Class '{class_name}' does not exist")
            return files
            
        # Find all files for this class
        for file_id, file_info in self.metadata['files'].items():
            if file_info['class'] == class_name:
                if not include_only_training or file_info['include_in_training']:
                    files.append(self.base_dir / file_id)
                    
        return files
        
    def get_all_files(self, include_only_training=True):
        """
        Get all files in the dataset.
        
        Args:
            include_only_training: Whether to include only training files
            
        Returns:
            Dictionary mapping class names to lists of file paths
        """
        all_files = {}
        
        for class_name in self.metadata['sound_classes'].keys():
            all_files[class_name] = self.get_files_by_class(class_name, include_only_training)
            
        return all_files
        
    def exclude_file(self, file_path, reason=""):
        """
        Exclude a file from training.
        
        Args:
            file_path: Path to the file
            reason: Reason for exclusion
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get relative path
            file_path = Path(file_path)
            if file_path.is_absolute():
                rel_path = file_path.relative_to(self.base_dir)
            else:
                rel_path = file_path
                
            file_id = str(rel_path)
            
            # Check if file exists in metadata
            if file_id not in self.metadata['files']:
                logging.error(f"File '{file_id}' not found in metadata")
                return False
                
            # Update metadata
            self.metadata['files'][file_id]['include_in_training'] = False
            self.metadata['files'][file_id]['exclusion_reason'] = reason
            
            # Save metadata
            self._save_metadata()
            
            return True
            
        except Exception as e:
            logging.error(f"Error excluding file '{file_path}': {str(e)}")
            return False
            
    def include_file(self, file_path):
        """
        Include a file in training (reverse of exclude).
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get relative path
            file_path = Path(file_path)
            if file_path.is_absolute():
                rel_path = file_path.relative_to(self.base_dir)
            else:
                rel_path = file_path
                
            file_id = str(rel_path)
            
            # Check if file exists in metadata
            if file_id not in self.metadata['files']:
                logging.error(f"File '{file_id}' not found in metadata")
                return False
                
            # Update metadata
            self.metadata['files'][file_id]['include_in_training'] = True
            if 'exclusion_reason' in self.metadata['files'][file_id]:
                del self.metadata['files'][file_id]['exclusion_reason']
            
            # Save metadata
            self._save_metadata()
            
            return True
            
        except Exception as e:
            logging.error(f"Error including file '{file_path}': {str(e)}")
            return False
    
    def update_file_status(self, file_path, status):
        """
        Update the preprocessing status of a file.
        
        Args:
            file_path: Path to the file
            status: New status (e.g., 'raw', 'training_sounds', 'preprocessed', 'augmented')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get relative path
            file_path = Path(file_path)
            if file_path.is_absolute():
                rel_path = file_path.relative_to(self.base_dir)
            else:
                rel_path = file_path
                
            file_id = str(rel_path)
            
            # Check if file exists in metadata
            if file_id not in self.metadata['files']:
                logging.error(f"File '{file_id}' not found in metadata")
                return False
                
            # Update metadata
            self.metadata['files'][file_id]['preprocessing_status'] = status
            
            # Save metadata
            self._save_metadata()
            
            return True
            
        except Exception as e:
            logging.error(f"Error updating file status '{file_path}': {str(e)}")
            return False
    
    def save_features(self, file_path, features, feature_type='unified'):
        """
        Save features for a file.
        
        Args:
            file_path: Path to the source audio file
            features: Features dictionary
            feature_type: Type of features ('unified' or model-specific)
            
        Returns:
            Path to the saved features if successful, None otherwise
        """
        try:
            # Get original file info
            file_path = Path(file_path)
            if file_path.is_absolute():
                rel_path = file_path.relative_to(self.base_dir)
            else:
                rel_path = file_path
            
            # Determine output directory
            if feature_type == 'unified':
                output_dir = self.unified_features_dir
            else:
                output_dir = self.model_specific_features_dir / feature_type
                output_dir.mkdir(exist_ok=True)
                
            # Create output filename
            output_filename = f"{rel_path.stem}_features.npz"
            output_path = output_dir / output_filename
            
            # Save features
            np.savez_compressed(output_path, **features)
            
            # Update metadata if available
            file_id = str(rel_path)
            if file_id in self.metadata['files']:
                if 'features' not in self.metadata['files'][file_id]:
                    self.metadata['files'][file_id]['features'] = {}
                    
                self.metadata['files'][file_id]['features'][feature_type] = {
                    'path': str(output_path.relative_to(self.base_dir)),
                    'created_at': datetime.now().isoformat()
                }
                
                # Save metadata
                self._save_metadata()
                
            return output_path
            
        except Exception as e:
            logging.error(f"Error saving features for '{file_path}': {str(e)}")
            return None
    
    def load_features(self, file_path, feature_type='unified'):
        """
        Load features for a file.
        
        Args:
            file_path: Path to the source audio file
            feature_type: Type of features ('unified' or model-specific)
            
        Returns:
            Features dictionary if successful, None otherwise
        """
        try:
            # Get original file info
            file_path = Path(file_path)
            if file_path.is_absolute():
                rel_path = file_path.relative_to(self.base_dir)
            else:
                rel_path = file_path
                
            file_id = str(rel_path)
            
            # Check if features exist in metadata
            if (file_id in self.metadata['files'] and 
                'features' in self.metadata['files'][file_id] and
                feature_type in self.metadata['files'][file_id]['features']):
                
                # Get feature path from metadata
                feature_path = self.base_dir / self.metadata['files'][file_id]['features'][feature_type]['path']
                
            else:
                # If not in metadata, construct path
                if feature_type == 'unified':
                    feature_path = self.unified_features_dir / f"{rel_path.stem}_features.npz"
                else:
                    feature_path = self.model_specific_features_dir / feature_type / f"{rel_path.stem}_features.npz"
            
            # Load features
            if not feature_path.exists():
                logging.error(f"Feature file '{feature_path}' not found")
                return None
                
            features = np.load(feature_path, allow_pickle=True)
            
            # Convert to dictionary
            features_dict = {key: features[key] for key in features.files}
            
            return features_dict
            
        except Exception as e:
            logging.error(f"Error loading features for '{file_path}': {str(e)}")
            return None
    
    def get_class_summary(self):
        """
        Get a summary of all classes in the dataset.
        
        Returns:
            DataFrame with class summary information
        """
        summary_data = []
        
        for class_name, class_info in self.metadata['sound_classes'].items():
            # Count files
            total_files = class_info['file_count']
            
            # Count training files
            training_files = 0
            for file_id, file_info in self.metadata['files'].items():
                if file_info['class'] == class_name and file_info['include_in_training']:
                    training_files += 1
                    
            # Add to summary
            summary_data.append({
                'class': class_name,
                'description': class_info['description'],
                'total_files': total_files,
                'training_files': training_files,
                'excluded_files': total_files - training_files
            })
            
        # Create DataFrame
        return pd.DataFrame(summary_data)
        
    def export_to_dataframe(self):
        """
        Export the entire dataset to a pandas DataFrame.
        
        Returns:
            DataFrame with all file information
        """
        data = []
        
        for file_id, file_info in self.metadata['files'].items():
            row = {
                'file_id': file_id,
                'class': file_info['class'],
                'added_at': file_info['added_at'],
                'include_in_training': file_info['include_in_training'],
                'preprocessing_status': file_info.get('preprocessing_status', 'unknown')
            }
            
            # Add feature information if available
            if 'features' in file_info:
                for feature_type, feature_info in file_info['features'].items():
                    row[f'features_{feature_type}'] = feature_info['path']
                    row[f'features_{feature_type}_created_at'] = feature_info['created_at']
                    
            data.append(row)
            
        # Create DataFrame
        return pd.DataFrame(data) 