import os
import json
import shutil
import logging
from datetime import datetime
from config import Config
import copy

class DictionaryService:
    """
    Service for managing sound dictionaries.
    Handles creating, updating, and deleting dictionaries and their sound classes.
    """
    
    def __init__(self, dictionaries_dir=None):
        """Initialize the dictionary service."""
        self.dictionaries_dir = dictionaries_dir or Config.DICTIONARIES_DIR
        logging.debug(f"Dictionary service using directory: {self.dictionaries_dir}")
        os.makedirs(self.dictionaries_dir, exist_ok=True)
        self.metadata_file = os.path.join(self.dictionaries_dir, 'metadata.json')
        logging.debug(f"Metadata file path: {self.metadata_file}")
        self._load_metadata()
    
    def _load_metadata(self):
        """Load dictionary metadata from file."""
        try:
            logging.debug(f"Loading metadata from: '{self.metadata_file}'")
            
            if os.path.exists(self.metadata_file):
                # Check if file is readable
                if not os.access(self.metadata_file, os.R_OK):
                    logging.error(f"No read permission for metadata file: '{self.metadata_file}'")
                    self.metadata = {"dictionaries": {}}
                    return
                
                # Check file size
                file_size = os.path.getsize(self.metadata_file)
                logging.debug(f"Metadata file size: {file_size} bytes")
                
                if file_size == 0:
                    logging.warning(f"Metadata file is empty: '{self.metadata_file}'")
                    self.metadata = {"dictionaries": {}}
                    return
                
                try:
                    with open(self.metadata_file, 'r') as f:
                        file_content = f.read()
                        logging.debug(f"Read {len(file_content)} bytes from metadata file")
                        
                        if not file_content.strip():
                            logging.warning("Metadata file is empty or contains only whitespace")
                            self.metadata = {"dictionaries": {}}
                            return
                            
                        self.metadata = json.loads(file_content)
                        
                        # Validate metadata structure
                        if not isinstance(self.metadata, dict):
                            logging.error(f"Loaded metadata is not a dictionary: {type(self.metadata)}")
                            self.metadata = {"dictionaries": {}}
                            return
                            
                        if "dictionaries" not in self.metadata:
                            logging.warning("Loaded metadata doesn't contain 'dictionaries' key, initializing it")
                            self.metadata["dictionaries"] = {}
                            
                        logging.debug(f"Successfully loaded metadata with {len(self.metadata.get('dictionaries', {}))} dictionaries")
                except json.JSONDecodeError as e:
                    logging.error(f"Invalid JSON in metadata file: {e}")
                    self.metadata = {"dictionaries": {}}
                except Exception as e:
                    logging.exception(f"Error loading dictionary metadata: {e}")
                    self.metadata = {"dictionaries": {}}
            else:
                logging.info(f"Metadata file doesn't exist, initializing empty metadata: '{self.metadata_file}'")
                self.metadata = {"dictionaries": {}}
        except Exception as e:
            logging.exception(f"Unexpected error loading metadata: {e}")
            self.metadata = {"dictionaries": {}}
    
    def _save_metadata(self):
        """Save dictionary metadata to file."""
        try:
            # Ensure the directory exists
            metadata_dir = os.path.dirname(self.metadata_file)
            os.makedirs(metadata_dir, exist_ok=True)
            logging.debug(f"Directory for metadata file exists: {metadata_dir}")
            
            # Check directory permissions
            if not os.access(metadata_dir, os.W_OK):
                logging.error(f"No write permission for metadata directory: {metadata_dir}")
                return False
            
            # Check if file exists and has write permissions
            if os.path.exists(self.metadata_file) and not os.access(self.metadata_file, os.W_OK):
                logging.error(f"No write permission for metadata file: {self.metadata_file}")
                return False
            
            # Validate metadata structure
            if not isinstance(self.metadata, dict):
                logging.error(f"Metadata is not a dictionary: {type(self.metadata)}")
                self.metadata = {"dictionaries": {}}
            
            if "dictionaries" not in self.metadata:
                logging.error("Metadata does not contain 'dictionaries' key, initializing it")
                self.metadata["dictionaries"] = {}
            
            # Save the metadata
            with open(self.metadata_file, 'w') as f:
                json_str = json.dumps(self.metadata, indent=2)
                f.write(json_str)
                logging.debug(f"Wrote {len(json_str)} bytes to metadata file")
            
            # Verify the file was written
            if os.path.exists(self.metadata_file):
                logging.debug(f"Metadata saved successfully to {self.metadata_file}")
                return True
            else:
                logging.error(f"Metadata file does not exist after writing: {self.metadata_file}")
                return False
        except Exception as e:
            logging.exception(f"Error saving dictionary metadata: {e}")
            return False
    
    def create_dictionary(self, name, description="", user_id=None):
        """
        Create a new dictionary.
        
        Args:
            name (str): Dictionary name
            description (str): Dictionary description
            user_id (str, optional): ID of the user creating the dictionary
            
        Returns:
            dict: Dictionary information or error
        """
        try:
            if not name:
                logging.error("Dictionary name is required")
                return {
                    "success": False,
                    "error": "Dictionary name is required"
                }
                
            logging.debug(f"Creating dictionary: name='{name}', description='{description}', user_id='{user_id}'")
            
            # Sanitize name for filesystem use
            safe_name = name.replace(' ', '_').lower()
            logging.debug(f"Sanitized name: '{safe_name}'")
            
            # Ensure metadata has been initialized
            if not hasattr(self, 'metadata') or not isinstance(self.metadata, dict):
                logging.error("Metadata not initialized correctly")
                self._load_metadata()
                
            if "dictionaries" not in self.metadata:
                logging.warning("Dictionaries key not found in metadata, initializing")
                self.metadata["dictionaries"] = {}
            
            # Check if dictionary already exists
            if safe_name in self.metadata["dictionaries"]:
                logging.warning(f"Dictionary '{name}' already exists")
                return {
                    "success": False,
                    "error": f"Dictionary '{name}' already exists"
                }
            
            # Create directory
            dict_path = os.path.join(self.dictionaries_dir, safe_name)
            logging.debug(f"Creating directory at: '{dict_path}'")
            
            # Check if directory already exists
            if os.path.exists(dict_path):
                logging.warning(f"Directory already exists at '{dict_path}'")
                
            # Check directory permissions
            if not os.access(self.dictionaries_dir, os.W_OK):
                logging.error(f"No write permission for dictionaries directory: '{self.dictionaries_dir}'")
                return {
                    "success": False,
                    "error": f"No permission to create dictionary in '{self.dictionaries_dir}'"
                }
                
            # Create the directory
            try:
                os.makedirs(dict_path, exist_ok=True)
                logging.debug(f"Directory created successfully at '{dict_path}'")
            except Exception as e:
                logging.exception(f"Failed to create directory: {e}")
                return {
                    "success": False,
                    "error": f"Failed to create dictionary directory: {e}"
                }
            
            # Create metadata
            timestamp = datetime.now().isoformat()
            dict_info = {
                "name": name,
                "description": description,
                "created_at": timestamp,
                "updated_at": timestamp,
                "created_by": user_id,
                "classes": [],
                "sample_count": 0,
                "path": dict_path
            }
            logging.debug(f"Dictionary metadata created: {dict_info}")
            
            # Add to metadata
            self.metadata["dictionaries"][safe_name] = dict_info
            logging.debug(f"Saving metadata to '{self.metadata_file}'")
            
            # Save metadata
            if self._save_metadata():
                logging.info(f"Dictionary '{name}' created successfully")
                return {
                    "success": True,
                    "dictionary": dict_info
                }
            else:
                logging.error(f"Failed to save metadata for dictionary '{name}'")
                
                # Check if directory was created but metadata failed
                if os.path.exists(dict_path):
                    try:
                        logging.warning(f"Cleaning up directory after metadata save failure: '{dict_path}'")
                        shutil.rmtree(dict_path)
                    except Exception as cleanup_error:
                        logging.error(f"Failed to clean up directory: {cleanup_error}")
                        
                return {
                    "success": False,
                    "error": "Failed to save dictionary metadata"
                }
        except Exception as e:
            logging.exception(f"Unexpected error creating dictionary '{name}': {e}")
            return {
                "success": False,
                "error": f"Unexpected error creating dictionary: {e}"
            }
    
    def get_dictionaries(self, user_id=None):
        """
        Get all dictionaries, optionally filtered by user.
        
        Args:
            user_id (str, optional): Filter by user ID
            
        Returns:
            list: List of dictionaries with enhanced class information
        """
        logging.debug(f"get_dictionaries called with user_id: {user_id}")
        logging.debug(f"Total dictionaries in metadata: {len(self.metadata.get('dictionaries', {}))}")
        
        # If metadata is empty or corrupted, initialize it
        if not isinstance(self.metadata, dict) or "dictionaries" not in self.metadata:
            logging.warning("Metadata is corrupted or missing, reinitializing")
            self.metadata = {"dictionaries": {}}
            self._save_metadata()
            return []
        
        # Get all dictionaries or filter by user
        if user_id is None:
            dictionaries = list(self.metadata["dictionaries"].values())
            logging.debug(f"Getting all {len(dictionaries)} dictionaries")
        else:
            # Filter by user
            dictionaries = [
                dict_info for dict_info in self.metadata["dictionaries"].values()
                if dict_info.get("created_by") == user_id
            ]
            logging.debug(f"Filtered to {len(dictionaries)} dictionaries for user_id: {user_id}")
        
        # For diagnostics, log all unique creator IDs
        creator_ids = set(dict_info.get("created_by") for dict_info in self.metadata["dictionaries"].values() 
                        if dict_info.get("created_by") is not None)
        logging.debug(f"Unique creator IDs in metadata: {creator_ids}")
        
        # Enhance each dictionary with class-level sample counts
        enhanced_dictionaries = []
        for dictionary in dictionaries:
            # Create a copy to avoid modifying the original metadata
            dict_copy = copy.deepcopy(dictionary)
            
            # Sync sample counts to ensure they're accurate (this updates the metadata)
            self.sync_dictionary_samples(dictionary['name'])
            
            # Get the fresh metadata
            safe_dict_name = dictionary['name'].replace(' ', '_').lower()
            fresh_dict_info = self.metadata["dictionaries"].get(safe_dict_name)
            
            # Update the dictionary copy with fresh data
            if fresh_dict_info:
                dict_copy['sample_count'] = fresh_dict_info['sample_count']
                dict_copy['updated_at'] = fresh_dict_info['updated_at']
            
            # Add class_details with sample counts
            dict_copy['class_details'] = []
            
            if 'classes' in dictionary and dictionary['classes']:
                for class_name in dictionary['classes']:
                    # Check samples in the dictionaries dir
                    class_path = os.path.join(self.dictionaries_dir, safe_dict_name, class_name)
                    sample_count = 0
                    
                    if os.path.exists(class_path) and os.path.isdir(class_path):
                        sample_count = len([f for f in os.listdir(class_path) 
                                          if os.path.isfile(os.path.join(class_path, f)) 
                                          and f.lower().endswith(('.wav', '.mp3', '.ogg'))])
                    
                    # Also check samples in the central sounds dir
                    central_class_path = os.path.join(Config.TRAINING_SOUNDS_DIR, class_name)
                    central_sample_count = 0
                    
                    if os.path.exists(central_class_path) and os.path.isdir(central_class_path):
                        central_sample_count = len([f for f in os.listdir(central_class_path) 
                                                 if os.path.isfile(os.path.join(central_class_path, f)) 
                                                 and f.lower().endswith(('.wav', '.mp3', '.ogg'))])
                    
                    # Use the maximum count (could be in either location)
                    effective_sample_count = max(sample_count, central_sample_count)
                    
                    dict_copy['class_details'].append({
                        'name': class_name,
                        'sample_count': effective_sample_count
                    })
            
            enhanced_dictionaries.append(dict_copy)
        
        return enhanced_dictionaries
    
    def get_dictionary(self, dict_name):
        """
        Get a dictionary by name.
        
        Args:
            dict_name (str): Dictionary name
            
        Returns:
            dict: Dictionary information or None
        """
        safe_name = dict_name.replace(' ', '_').lower()
        
        # Get the base dictionary information
        dict_info = self.metadata["dictionaries"].get(safe_name)
        
        if dict_info:
            # Let's sync the sample count with actual files on disk
            self.sync_dictionary_samples(dict_name)
            
            # Get the updated information
            dict_info = self.metadata["dictionaries"].get(safe_name)
            
        return dict_info
    
    def sync_dictionary_samples(self, dict_name):
        """
        Sync the sample count for a dictionary by counting samples on disk.
        
        Args:
            dict_name (str): Dictionary name
            
        Returns:
            dict: Updated dictionary information
        """
        safe_dict_name = dict_name.replace(' ', '_').lower()
        
        # Check if dictionary exists
        if safe_dict_name not in self.metadata["dictionaries"]:
            return {
                "success": False,
                "error": f"Dictionary '{dict_name}' does not exist"
            }
        
        # Get dictionary info
        dict_info = self.metadata["dictionaries"][safe_dict_name]
        
        # Count samples for each class
        total_samples = 0
        class_counts = {}
        
        for class_name in dict_info.get("classes", []):
            # Get sample count from central sounds directory
            central_class_path = os.path.join(Config.TRAINING_SOUNDS_DIR, class_name)
            if os.path.exists(central_class_path):
                samples = [f for f in os.listdir(central_class_path) if f.lower().endswith('.wav')]
                class_counts[class_name] = len(samples)
                total_samples += len(samples)
        
        # Update metadata
        dict_info["sample_count"] = total_samples
        dict_info["class_counts"] = class_counts
        dict_info["updated_at"] = datetime.now().isoformat()
        
        # Save metadata
        if self._save_metadata():
            return {
                "success": True,
                "dictionary": {
                    "name": dict_info["name"],
                    "sample_count": total_samples,
                    "class_counts": class_counts
                }
            }
        else:
            return {
                "success": False,
                "error": "Failed to save metadata"
            }
    
    def add_class(self, dict_name, class_name):
        """
        Add a new sound class to a dictionary.
        
        Args:
            dict_name (str): Dictionary name
            class_name (str): Class name
            
        Returns:
            dict: Success/error information
        """
        logging.debug(f"Adding class '{class_name}' to dictionary '{dict_name}'")
        
        safe_dict_name = dict_name.replace(' ', '_').lower()
        logging.error(f"Safe dictionary name: {safe_dict_name}")
        
        # Check if dictionary exists
        if safe_dict_name not in self.metadata["dictionaries"]:
            logging.error(f"Dictionary '{dict_name}' (safe name: {safe_dict_name}) does not exist")
            logging.error(f"Available dictionaries: {list(self.metadata['dictionaries'].keys())}")
            return {
                "success": False,
                "error": f"Dictionary '{dict_name}' does not exist"
            }
        
        # Sanitize class name
        safe_class_name = class_name.replace(' ', '_').lower()
        logging.debug(f"Safe class name: {safe_class_name}")
        
        # Check if class already exists
        dict_info = self.metadata["dictionaries"][safe_dict_name]
        logging.debug(f"Dictionary info: {dict_info}")
        logging.debug(f"Existing classes: {dict_info.get('classes', [])}")
        
        if safe_class_name in dict_info["classes"]:
            logging.warning(f"Class '{class_name}' already exists in dictionary '{dict_name}'")
            return {
                "success": False,
                "error": f"Class '{class_name}' already exists in dictionary '{dict_name}'"
            }
        
        # Create class directory
        class_path = os.path.join(self.dictionaries_dir, safe_dict_name, safe_class_name)
        logging.debug(f"Creating class directory at: {class_path}")
        try:
            os.makedirs(class_path, exist_ok=True)
            logging.debug(f"Class directory created successfully")
        except Exception as e:
            logging.error(f"Failed to create class directory: {e}")
            return {
                "success": False,
                "error": f"Failed to create class directory: {e}"
            }
        
        # Update metadata
        dict_info["classes"].append(safe_class_name)
        dict_info["updated_at"] = datetime.now().isoformat()
        logging.debug(f"Updated dictionary metadata with new class: {safe_class_name}")
        logging.debug(f"Updated dictionary classes: {dict_info['classes']}")
        
        # Save metadata directly to the file system for immediate availability
        try:
            # First, verify that the metadata is properly updated in our in-memory dictionary
            updated_classes = self.metadata["dictionaries"][safe_dict_name]["classes"]
            logging.debug(f"Verified classes in metadata before saving: {updated_classes}")
            
            # Now save the metadata
            if self._save_metadata():
                logging.info(f"Class '{class_name}' added successfully to dictionary '{dict_name}'")
                
                # Double-check the metadata after saving
                self._load_metadata()  # Reload from disk
                after_classes = self.metadata["dictionaries"][safe_dict_name].get("classes", [])
                logging.debug(f"Classes after reload: {after_classes}")
                
                return {
                    "success": True,
                    "class": {
                        "name": class_name,
                        "safe_name": safe_class_name,
                        "path": class_path
                    }
                }
            else:
                logging.error(f"Failed to save metadata after adding class '{class_name}'")
                return {
                    "success": False,
                    "error": "Failed to save class metadata"
                }
        except Exception as e:
            logging.exception(f"Error updating metadata: {e}")
            return {
                "success": False,
                "error": f"Error updating metadata: {str(e)}"
            }
    
    def add_sample(self, dict_name, class_name, file_path, sample_name=None):
        """
        Add a sound sample to a class.
        
        Args:
            dict_name (str): Dictionary name
            class_name (str): Class name
            file_path (str): Path to audio file
            sample_name (str, optional): Custom name for the sample
            
        Returns:
            dict: Success/error information
        """
        safe_dict_name = dict_name.replace(' ', '_').lower()
        safe_class_name = class_name.replace(' ', '_').lower()
        
        # Check if dictionary exists
        dict_info = self.metadata["dictionaries"].get(safe_dict_name)
        if not dict_info:
            return {
                "success": False,
                "error": f"Dictionary '{dict_name}' does not exist"
            }
        
        # Check if class exists in this dictionary
        if safe_class_name not in dict_info["classes"]:
            return {
                "success": False,
                "error": f"Class '{class_name}' does not exist in dictionary '{dict_name}'"
            }
        
        # Generate sample name if not provided
        if not sample_name:
            sample_name = os.path.basename(file_path)
        
        # Ensure sample has .wav extension
        if not sample_name.lower().endswith('.wav'):
            sample_name += '.wav'
        
        # Make sure both the central sounds directory and dictionary-specific class directory exist
        # 1. Central sounds directory (using TRAINING_SOUNDS_DIR instead of SOUNDS_DIR)
        os.makedirs(Config.TRAINING_SOUNDS_DIR, exist_ok=True)
        central_class_path = os.path.join(Config.TRAINING_SOUNDS_DIR, safe_class_name)
        os.makedirs(central_class_path, exist_ok=True)
        
        # 2. Dictionary-specific class directory
        dict_class_path = os.path.join(self.dictionaries_dir, safe_dict_name, safe_class_name)
        os.makedirs(dict_class_path, exist_ok=True)
        
        # Copy file to both locations
        try:
            # Copy to central sounds directory
            central_target_path = os.path.join(central_class_path, sample_name)
            shutil.copy2(file_path, central_target_path)
            
            # Copy to dictionary-specific class directory
            dict_target_path = os.path.join(dict_class_path, sample_name)
            shutil.copy2(file_path, dict_target_path)
            
            logging.debug(f"Sample added to central path: {central_target_path}")
            logging.debug(f"Sample added to dictionary path: {dict_target_path}")
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to copy sample: {e}"
            }
        
        # Update metadata and sync sample counts
        self.sync_dictionary_samples(dict_name)
        
        return {
            "success": True,
            "sample": {
                "name": sample_name,
                "central_path": central_target_path,
                "dict_path": dict_target_path,
                "dictionary": dict_name,
                "class": class_name
            }
        }
    
    def get_class_samples(self, dict_name, class_name):
        """
        Get all samples in a class.
        
        Args:
            dict_name (str): Dictionary name
            class_name (str): Class name
            
        Returns:
            list: List of sample paths
        """
        safe_dict_name = dict_name.replace(' ', '_').lower()
        safe_class_name = class_name.replace(' ', '_').lower()
        
        # Check if dictionary and class exist
        class_path = os.path.join(self.dictionaries_dir, safe_dict_name, safe_class_name)
        if not os.path.exists(class_path):
            return []
        
        # Get all .wav files
        samples = []
        for filename in os.listdir(class_path):
            if filename.lower().endswith('.wav'):
                sample_path = os.path.join(class_path, filename)
                samples.append({
                    "name": filename,
                    "path": sample_path,
                    "size": os.path.getsize(sample_path)
                })
        
        return samples
    
    def delete_dictionary(self, dict_name):
        """
        Delete a dictionary and all its contents.
        
        Args:
            dict_name (str): Dictionary name
            
        Returns:
            dict: Success/error information
        """
        safe_name = dict_name.replace(' ', '_').lower()
        
        # Check if dictionary exists
        if safe_name not in self.metadata["dictionaries"]:
            return {
                "success": False,
                "error": f"Dictionary '{dict_name}' does not exist"
            }
        
        # Delete directory
        dict_path = os.path.join(self.dictionaries_dir, safe_name)
        try:
            shutil.rmtree(dict_path)
        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to delete dictionary directory: {e}"
            }
        
        # Remove from metadata
        del self.metadata["dictionaries"][safe_name]
        
        if self._save_metadata():
            return {
                "success": True,
                "message": f"Dictionary '{dict_name}' deleted successfully"
            }
        else:
            return {
                "success": False,
                "error": "Failed to update metadata"
            }
    
    def delete_class(self, dict_name, class_name):
        """
        Delete a sound class from a dictionary. This only removes the reference to the class,
        not the actual sound files.
        
        Args:
            dict_name (str): Dictionary name
            class_name (str): Class name
            
        Returns:
            dict: Success/error information
        """
        logging.debug(f"Deleting class '{class_name}' from dictionary '{dict_name}'")
        
        safe_dict_name = dict_name.replace(' ', '_').lower()
        safe_class_name = class_name.replace(' ', '_').lower()
        
        # Check if dictionary exists
        if safe_dict_name not in self.metadata["dictionaries"]:
            logging.error(f"Dictionary '{dict_name}' does not exist")
            return {
                "success": False,
                "error": f"Dictionary '{dict_name}' does not exist"
            }
        
        # Check if class exists
        dict_info = self.metadata["dictionaries"][safe_dict_name]
        if safe_class_name not in dict_info["classes"]:
            logging.error(f"Class '{class_name}' does not exist in dictionary '{dict_name}'")
            return {
                "success": False,
                "error": f"Class '{class_name}' does not exist in dictionary '{dict_name}'"
            }
        
        # Get sample count for this class
        sample_count_reduction = 0
        class_path = os.path.join(Config.TRAINING_SOUNDS_DIR, safe_class_name)
        if os.path.exists(class_path):
            sample_files = [f for f in os.listdir(class_path) if f.lower().endswith('.wav')]
            sample_count_reduction = len(sample_files)
        
        # Remove class from dictionary's class list
        dict_info["classes"].remove(safe_class_name)
        dict_info["updated_at"] = datetime.now().isoformat()
        dict_info["sample_count"] = max(0, dict_info["sample_count"] - sample_count_reduction)
        
        # Save metadata
        if self._save_metadata():
            logging.info(f"Class '{class_name}' removed from dictionary '{dict_name}'")
            return {
                "success": True,
                "message": f"Class '{class_name}' removed from dictionary '{dict_name}'"
            }
        else:
            logging.error(f"Failed to save metadata after removing class '{class_name}'")
            return {
                "success": False,
                "error": "Failed to save metadata after removing class"
            }
    
    def get_samples(self, dict_name, class_name):
        """
        Get all samples for a class.
        
        Args:
            dict_name (str): Dictionary name
            class_name (str): Class name
            
        Returns:
            dict: Success/error information with samples list
        """
        safe_dict_name = dict_name.replace(' ', '_').lower()
        safe_class_name = class_name.replace(' ', '_').lower()
        
        # Check if dictionary exists
        if safe_dict_name not in self.metadata["dictionaries"]:
            return {
                "success": False,
                "error": f"Dictionary '{dict_name}' does not exist"
            }
        
        # Check if class exists in this dictionary
        dict_info = self.metadata["dictionaries"][safe_dict_name]
        if safe_class_name not in dict_info["classes"]:
            return {
                "success": False,
                "error": f"Class '{class_name}' does not exist in dictionary '{dict_name}'"
            }
        
        # Get samples from the central sounds directory
        class_path = os.path.join(Config.TRAINING_SOUNDS_DIR, safe_class_name)
        if not os.path.exists(class_path):
            return {
                "success": True,
                "samples": []
            }
        
        # Get all .wav files
        samples = []
        for filename in os.listdir(class_path):
            if filename.lower().endswith('.wav'):
                sample_path = os.path.join(class_path, filename)
                samples.append({
                    "name": filename,
                    "path": sample_path,
                    "size": os.path.getsize(sample_path)
                })
        
        return {
            "success": True,
            "samples": samples
        }

    def get_sound_classes(self):
        """
        Get all available sound classes from the training sounds directory.
        
        Returns:
            list: List of sound class names
        """
        sound_classes = []
        
        try:
            # Check if the training sounds directory exists
            if os.path.exists(Config.TRAINING_SOUNDS_DIR):
                # Get all subdirectories in the sounds directory
                for item in os.listdir(Config.TRAINING_SOUNDS_DIR):
                    item_path = os.path.join(Config.TRAINING_SOUNDS_DIR, item)
                    if os.path.isdir(item_path):
                        # Count WAV files in the directory
                        wav_files = [f for f in os.listdir(item_path) if f.lower().endswith('.wav')]
                        sound_classes.append({
                            "name": item,
                            "sample_count": len(wav_files)
                        })
        except Exception as e:
            logging.error(f"Error getting sound classes: {e}")
        
        return sound_classes
