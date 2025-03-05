import os
import json
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
import logging
from datetime import datetime

class SoundDatasetManager:
    """
    Manages a dataset of sound files with metadata for training and evaluation.
    
    This class provides functionality to:
    - Track sound files and their metadata
    - Mark files for inclusion/exclusion in training
    - Add new files to the dataset
    - Export dataset for model training
    """
    
    def __init__(self, data_dir="data", sounds_dir="sounds/training_sounds", 
                 metadata_file="dataset_metadata.json", feature_dir="saved_features"):
        """
        Initialize the dataset manager.
        
        Args:
            data_dir: Base directory for all data
            sounds_dir: Directory containing sound files, relative to data_dir
            metadata_file: File to store dataset metadata
            feature_dir: Directory for extracted features
        """
        self.data_dir = Path(data_dir)
        self.sounds_dir = self.data_dir / sounds_dir
        self.metadata_path = self.data_dir / metadata_file
        self.feature_dir = self.data_dir / feature_dir
        
        # Create directories if they don't exist
        self.sounds_dir.mkdir(parents=True, exist_ok=True)
        self.feature_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize metadata
        self.metadata = self._load_metadata()
        
    def _load_metadata(self):
        """Load metadata from file or initialize if it doesn't exist."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Error loading metadata: {e}")
                return self._initialize_metadata()
        else:
            return self._initialize_metadata()
    
    def _initialize_metadata(self):
        """Create initial metadata structure."""
        metadata = {
            "files": {},
            "classes": {},
            "dataset_info": {
                "total_files": 0,
                "included_files": 0,
                "excluded_files": 0,
                "last_updated": None
            }
        }
        return metadata
    
    def _save_metadata(self):
        """Save metadata to file."""
        # Update dataset stats
        included = sum(1 for f in self.metadata["files"].values() if f.get("include_in_training", True))
        excluded = sum(1 for f in self.metadata["files"].values() if not f.get("include_in_training", True))
        
        self.metadata["dataset_info"]["total_files"] = len(self.metadata["files"])
        self.metadata["dataset_info"]["included_files"] = included
        self.metadata["dataset_info"]["excluded_files"] = excluded
        self.metadata["dataset_info"]["last_updated"] = datetime.now().isoformat()
        
        # Save to file
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
            
    def scan_directory(self):
        """Scan the sounds directory and update metadata."""
        file_count = 0
        new_files = 0
        
        # Get all class directories
        class_dirs = [d for d in os.listdir(self.sounds_dir) 
                     if os.path.isdir(os.path.join(self.sounds_dir, d))]
        
        for class_name in class_dirs:
            class_dir = os.path.join(self.sounds_dir, class_name)
            
            # Initialize class in metadata if not exists
            if class_name not in self.metadata["classes"]:
                self.metadata["classes"][class_name] = {
                    "file_count": 0,
                    "included_count": 0
                }
            
            # Get all WAV files in this class
            wav_files = [f for f in os.listdir(class_dir) 
                        if f.lower().endswith('.wav')]
            
            for filename in wav_files:
                file_path = os.path.join(class_dir, filename)
                rel_path = os.path.join(class_name, filename)
                file_count += 1
                
                # Add file to metadata if not already present
                if rel_path not in self.metadata["files"]:
                    self.metadata["files"][rel_path] = {
                        "class": class_name,
                        "filename": filename,
                        "include_in_training": True,
                        "quality_rating": None,
                        "notes": "",
                        "added_date": datetime.now().isoformat()
                    }
                    new_files += 1
            
            # Update class file count
            included = sum(1 for f, meta in self.metadata["files"].items() 
                         if meta["class"] == class_name and meta.get("include_in_training", True))
            
            self.metadata["classes"][class_name]["file_count"] = len(wav_files)
            self.metadata["classes"][class_name]["included_count"] = included
        
        # Save updated metadata
        self._save_metadata()
        
        return {
            "total_files": file_count,
            "new_files": new_files
        }
    
    def exclude_file(self, file_path, reason=""):
        """
        Exclude a file from training.
        
        Args:
            file_path: Path to the file relative to the class directory
            reason: Reason for exclusion
        """
        if file_path in self.metadata["files"]:
            self.metadata["files"][file_path]["include_in_training"] = False
            self.metadata["files"][file_path]["notes"] = reason
            
            # Update class stats
            class_name = self.metadata["files"][file_path]["class"]
            included = sum(1 for f, meta in self.metadata["files"].items() 
                         if meta["class"] == class_name and meta.get("include_in_training", True))
            self.metadata["classes"][class_name]["included_count"] = included
            
            self._save_metadata()
            return True
        return False
    
    def include_file(self, file_path):
        """Re-include a previously excluded file for training."""
        if file_path in self.metadata["files"]:
            self.metadata["files"][file_path]["include_in_training"] = True
            
            # Update class stats
            class_name = self.metadata["files"][file_path]["class"]
            included = sum(1 for f, meta in self.metadata["files"].items() 
                         if meta["class"] == class_name and meta.get("include_in_training", True))
            self.metadata["classes"][class_name]["included_count"] = included
            
            self._save_metadata()
            return True
        return False
    
    def add_file(self, source_path, class_name, filename=None, include_in_training=True):
        """
        Add a new file to the dataset.
        
        Args:
            source_path: Path to the source file
            class_name: Class to add the file to
            filename: Custom filename (or use original if None)
            include_in_training: Whether to include in training
        """
        if filename is None:
            filename = os.path.basename(source_path)
        
        # Ensure class directory exists
        class_dir = os.path.join(self.sounds_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Copy file to dataset
        dest_path = os.path.join(class_dir, filename)
        shutil.copy2(source_path, dest_path)
        
        # Add to metadata
        rel_path = os.path.join(class_name, filename)
        self.metadata["files"][rel_path] = {
            "class": class_name,
            "filename": filename,
            "include_in_training": include_in_training,
            "quality_rating": None,
            "notes": "",
            "added_date": datetime.now().isoformat()
        }
        
        # Update class info
        if class_name not in self.metadata["classes"]:
            self.metadata["classes"][class_name] = {
                "file_count": 1,
                "included_count": 1 if include_in_training else 0
            }
        else:
            self.metadata["classes"][class_name]["file_count"] += 1
            if include_in_training:
                self.metadata["classes"][class_name]["included_count"] += 1
        
        self._save_metadata()
        return rel_path
    
    def set_quality_rating(self, file_path, rating):
        """
        Set quality rating for a file (e.g., 1-5 stars).
        
        Args:
            file_path: Path to the file relative to class directory
            rating: Quality rating (e.g., 1-5)
        """
        if file_path in self.metadata["files"]:
            self.metadata["files"][file_path]["quality_rating"] = rating
            self._save_metadata()
            return True
        return False
    
    def get_training_files(self):
        """Get list of files to include in training."""
        return [os.path.join(self.sounds_dir, path) 
                for path, meta in self.metadata["files"].items() 
                if meta.get("include_in_training", True)]
    
    def get_class_files(self, class_name, included_only=True):
        """Get list of files for a specific class."""
        return [os.path.join(self.sounds_dir, path) 
                for path, meta in self.metadata["files"].items() 
                if meta["class"] == class_name and 
                (not included_only or meta.get("include_in_training", True))]
    
    def get_dataset_summary(self):
        """Get summary statistics about the dataset."""
        return {
            "total_files": self.metadata["dataset_info"]["total_files"],
            "included_files": self.metadata["dataset_info"]["included_files"],
            "excluded_files": self.metadata["dataset_info"]["excluded_files"],
            "classes": {
                name: {
                    "total": info["file_count"],
                    "included": info["included_count"]
                } for name, info in self.metadata["classes"].items()
            }
        }
    
    def export_to_dataframe(self):
        """Export metadata to pandas DataFrame for analysis."""
        records = []
        for rel_path, meta in self.metadata["files"].items():
            record = {
                "path": rel_path,
                "class": meta["class"],
                "filename": meta["filename"],
                "include_in_training": meta.get("include_in_training", True),
                "quality_rating": meta.get("quality_rating"),
                "notes": meta.get("notes", ""),
                "added_date": meta.get("added_date")
            }
            records.append(record)
        
        return pd.DataFrame(records)
    
    def preprocess_included_files(self, extractor, force_reprocess=False):
        """
        Process only included files with the feature extractor.
        
        Args:
            extractor: FeatureExtractor instance
            force_reprocess: Whether to force reprocessing of already cached files
        """
        from tqdm import tqdm
        
        # Get list of files to process
        files_to_process = [os.path.join(self.sounds_dir, path) 
                           for path, meta in self.metadata["files"].items() 
                           if meta.get("include_in_training", True)]
        
        # Process files
        results = {"processed": 0, "skipped": 0, "errors": 0}
        
        for file_path in tqdm(files_to_process, desc="Processing files"):
            try:
                if not os.path.exists(file_path):
                    print(f"File not found: {file_path}")
                    results["errors"] += 1
                    continue
                
                # Check if already in cache (unless forcing reprocess)
                if not force_reprocess and extractor._check_cache(extractor._hash_file_path(file_path)):
                    results["skipped"] += 1
                    continue
                
                # Extract features
                features = extractor.extract_features(file_path, is_file=True)
                if features is not None:
                    results["processed"] += 1
                else:
                    results["errors"] += 1
            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
                results["errors"] += 1
        
        return results

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    manager = SoundDatasetManager()
    scan_results = manager.scan_directory()
    print(f"Found {scan_results['total_files']} files ({scan_results['new_files']} new)")
    print(manager.get_dataset_summary()) 