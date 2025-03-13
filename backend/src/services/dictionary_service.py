import os
import json
import shutil
import logging
from datetime import datetime
from backend.config import Config
import copy

class DictionaryService:
    """
    Service for managing sound dictionaries.
    Handles creating, updating, and deleting dictionaries and their sound classes.
    """
    
    def __init__(self, dictionaries_dir=None):
        """Initialize the dictionary service."""
        # Directory for actual sound data is data/sounds/training_sounds
        self.training_sounds_dir = Config.TRAINING_SOUNDS_DIR
        logging.debug(f"Dictionary service using training sounds directory: {self.training_sounds_dir}")
        
        # Directory for dictionary metadata is data/dictionaries
        self.dictionaries_dir = Config.DICTIONARIES_DIR  # Use the centralized path from Config
        logging.debug(f"Dictionary service using metadata directory: {self.dictionaries_dir}")
        
        # Ensure both directories exist
        os.makedirs(self.training_sounds_dir, exist_ok=True)
        os.makedirs(self.dictionaries_dir, exist_ok=True)
        
        # Path to the consolidated dictionaries.json file
        self.dictionaries_file = os.path.join(self.dictionaries_dir, 'dictionaries.json')
        logging.debug(f"Dictionaries file path: {self.dictionaries_file}")
        
        # Load dictionaries
        self._load_dictionaries()
    
    def _load_dictionaries(self):
        """Load dictionaries from the consolidated JSON file."""
        try:
            logging.debug(f"Loading dictionaries from: '{self.dictionaries_file}'")
            
            if os.path.exists(self.dictionaries_file):
                # Check if file is readable
                if not os.access(self.dictionaries_file, os.R_OK):
                    logging.error(f"No read permission for dictionaries file: '{self.dictionaries_file}'")
                    self.dictionaries = {"dictionaries": {}, "active_dictionary": None}
                    return
                
                # Check file size
                file_size = os.path.getsize(self.dictionaries_file)
                logging.debug(f"Dictionaries file size: {file_size} bytes")
                
                if file_size == 0:
                    logging.warning(f"Dictionaries file is empty: '{self.dictionaries_file}'")
                    self.dictionaries = {"dictionaries": {}, "active_dictionary": None}
                    return
                
                try:
                    with open(self.dictionaries_file, 'r') as f:
                        file_content = f.read()
                        logging.debug(f"Read {len(file_content)} bytes from dictionaries file")
                        
                        if not file_content.strip():
                            logging.warning("Dictionaries file is empty or contains only whitespace")
                            self.dictionaries = {"dictionaries": {}, "active_dictionary": None}
                            return
                            
                        self.dictionaries = json.loads(file_content)
                        
                        # Validate dictionaries structure
                        if not isinstance(self.dictionaries, dict):
                            logging.error(f"Loaded dictionaries is not a dictionary: {type(self.dictionaries)}")
                            self.dictionaries = {"dictionaries": {}, "active_dictionary": None}
                            return
                            
                        if "dictionaries" not in self.dictionaries:
                            logging.warning("Loaded dictionaries doesn't contain 'dictionaries' key, initializing it")
                            self.dictionaries["dictionaries"] = {}
                            
                        # If the dictionaries is a list, convert it to a dictionary with name as key
                        if isinstance(self.dictionaries["dictionaries"], list):
                            dict_obj = {}
                            for dictionary in self.dictionaries["dictionaries"]:
                                name = dictionary.get('name')
                                if name:
                                    dict_obj[name] = dictionary
                            self.dictionaries["dictionaries"] = dict_obj
                            
                        logging.debug(f"Successfully loaded dictionaries with {len(self.dictionaries.get('dictionaries', {}))} dictionaries")
                except json.JSONDecodeError as e:
                    logging.error(f"Invalid JSON in dictionaries file: {e}")
                    self.dictionaries = {"dictionaries": {}, "active_dictionary": None}
                except Exception as e:
                    logging.exception(f"Error loading dictionaries: {e}")
                    self.dictionaries = {"dictionaries": {}, "active_dictionary": None}
            else:
                logging.info(f"Dictionaries file doesn't exist, initializing empty dictionaries: '{self.dictionaries_file}'")
                self.dictionaries = {"dictionaries": {}, "active_dictionary": None}
        except Exception as e:
            logging.exception(f"Unexpected error loading dictionaries: {e}")
            self.dictionaries = {"dictionaries": {}, "active_dictionary": None}
    
    def _save_dictionaries(self):
        """Save dictionaries to the consolidated JSON file."""
        try:
            logging.info(f"=== SAVING DICTIONARIES STARTED ===")
            # Ensure the directory exists
            dictionaries_dir = os.path.dirname(self.dictionaries_file)
            os.makedirs(dictionaries_dir, exist_ok=True)
            logging.info(f"Directory for dictionaries file exists: {dictionaries_dir}")
            
            # Check directory permissions
            if not os.access(dictionaries_dir, os.W_OK):
                logging.error(f"No write permission for dictionaries directory: {dictionaries_dir}")
                return False
            
            # Check if file exists and has write permissions
            if os.path.exists(self.dictionaries_file) and not os.access(self.dictionaries_file, os.W_OK):
                logging.error(f"No write permission for dictionaries file: {self.dictionaries_file}")
                return False
            
            # Validate dictionaries structure
            if not isinstance(self.dictionaries, dict):
                logging.error(f"Dictionaries is not a dictionary: {type(self.dictionaries)}")
                return False
                
            if "dictionaries" not in self.dictionaries:
                logging.error("No 'dictionaries' key in dictionaries")
                return False
                
            if not isinstance(self.dictionaries["dictionaries"], dict):
                logging.error(f"Dictionaries is not a dictionary object: {type(self.dictionaries['dictionaries'])}")
                return False
            
            logging.info(f"Dictionaries validation passed. Writing to file: {self.dictionaries_file}")
            logging.info(f"Dictionary names in dictionaries: {list(self.dictionaries['dictionaries'].keys())}")
            
            # Create a backup of the current file if it exists
            if os.path.exists(self.dictionaries_file):
                backup_file = f"{self.dictionaries_file}.bak"
                try:
                    shutil.copy2(self.dictionaries_file, backup_file)
                    logging.info(f"Created backup of dictionaries file: {backup_file}")
                except Exception as e:
                    logging.warning(f"Failed to create backup: {e}")
            
            # Write dictionaries to file
            try:
                with open(self.dictionaries_file, 'w') as f:
                    json.dump(self.dictionaries, f, indent=4)
                logging.info(f"Successfully wrote dictionaries to file: {self.dictionaries_file}")
                
                # Verify file exists and has content
                if os.path.exists(self.dictionaries_file) and os.path.getsize(self.dictionaries_file) > 0:
                    logging.info(f"Verification passed: file exists and has content")
                    return True
                else:
                    logging.error(f"Verification failed: file missing or empty")
                    return False
            except Exception as e:
                logging.exception(f"Error writing dictionaries to file: {e}")
                return False
        except Exception as e:
            logging.exception(f"Error saving dictionaries: {e}")
            return False
    
    def create_dictionary(self, name, description="", user_id=None):
        """
        Create a new dictionary.
        
        Args:
            name (str): Dictionary name
            description (str, optional): Dictionary description
            user_id (str, optional): User ID of the creator
            
        Returns:
            dict: Result with success flag and dictionary information
        """
        try:
            logging.info(f"Creating dictionary: name='{name}', description='{description}', user_id='{user_id}'")
            
            if not name:
                logging.error("Dictionary name is required")
                return {
                    "success": False,
                    "error": "Dictionary name is required"
                }
            
            # Sanitize the dictionary name for file system use
            safe_name = name.replace(' ', '_').lower()
            logging.info(f"Sanitized name: '{safe_name}'")
            
            # Ensure dictionaries has been initialized
            if not hasattr(self, 'dictionaries') or not isinstance(self.dictionaries, dict):
                logging.error("Dictionaries not initialized correctly")
                self._load_dictionaries()
                
            if "dictionaries" not in self.dictionaries:
                logging.warning("Dictionaries key not found in dictionaries, initializing")
                self.dictionaries["dictionaries"] = {}
            
            # Check if dictionary already exists
            if name in self.dictionaries["dictionaries"]:
                logging.warning(f"Dictionary '{name}' already exists")
                return {
                    "success": False,
                    "error": f"Dictionary '{name}' already exists"
                }
            
            # Create dictionary metadata
            timestamp = datetime.now().isoformat()
            dict_info = {
                "id": safe_name,
                "name": name,
                "description": description,
                "created_at": timestamp,
                "updated_at": timestamp,
                "created_by": user_id,
                "classes": [],
                "sounds": [],  # For backward compatibility
                "sample_count": 0
            }
            logging.debug(f"Dictionary info created: {dict_info}")
            
            # Add to dictionaries dictionary
            self.dictionaries["dictionaries"][name] = dict_info
            logging.debug(f"Saving dictionaries to '{self.dictionaries_file}'")
            
            # Save dictionaries
            logging.info(f"Saving dictionaries with new dictionary '{safe_name}'")
            if self._save_dictionaries():
                logging.info(f"Dictionary '{name}' created successfully")
                return {
                    "success": True,
                    "dictionary": dict_info
                }
            else:
                logging.error(f"Failed to save dictionaries for dictionary '{name}'")
                return {
                    "success": False,
                    "error": "Failed to save dictionary data"
                }
        except Exception as e:
            logging.exception(f"Error creating dictionary: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_dictionaries(self, user_id=None):
        """
        Get a list of dictionaries.
        
        Args:
            user_id (str, optional): Filter dictionaries by user ID (currently ignored)
            
        Returns:
            list: List of dictionaries with enhanced class information
        """
        logging.info(f"=== GET DICTIONARIES CALLED ===")
        logging.info(f"get_dictionaries called with user_id: {user_id}, but returning all dictionaries")
        
        try:
            if not isinstance(self.dictionaries, dict) or "dictionaries" not in self.dictionaries:
                logging.warning("Dictionaries is corrupted or missing, reinitializing")
                self.dictionaries = {"dictionaries": {}, "active_dictionary": None}
                self._save_dictionaries()
                return []
                
            # Get the dictionaries object
            dictionaries_obj = self.dictionaries.get('dictionaries', {})
            logging.info(f"Total dictionaries in object: {len(dictionaries_obj)}")
            
            # Convert to list for processing - always return all dictionaries
            dictionaries_list = []
            for name, dict_data in dictionaries_obj.items():
                dictionaries_list.append(dict_data)
            
            dictionaries = dictionaries_list
            logging.info(f"Returning all {len(dictionaries)} dictionaries")
            
            # Enhance each dictionary with class-level sample counts
            enhanced_dictionaries = []
            for dictionary in dictionaries:
                # Create a copy to avoid modifying the original
                dict_copy = copy.deepcopy(dictionary)
                
                # Ensure classes field exists
                if 'classes' not in dict_copy and 'sounds' in dict_copy:
                    dict_copy['classes'] = dict_copy['sounds']
                elif 'classes' not in dict_copy:
                    dict_copy['classes'] = []
                
                # Get class details (sample counts, etc.)
                if dict_copy['classes']:
                    class_details = []
                    total_sample_count = 0  # For recalculating the actual sample count
                    
                    for class_name in dict_copy['classes']:
                        # Get class path in the training sounds directory
                        class_path = os.path.join(self.training_sounds_dir, class_name)
                        sample_count = 0
                        
                        # Count samples in the class directory
                        if os.path.exists(class_path) and os.path.isdir(class_path):
                            samples = [f for f in os.listdir(class_path) if f.endswith('.wav') or f.endswith('.mp3')]
                            sample_count = len(samples)
                            total_sample_count += sample_count
                            
                        class_details.append({
                            'name': class_name,
                            'sample_count': sample_count
                        })
                    
                    dict_copy['class_details'] = class_details
                    # Update the sample_count field with the accurate count
                    dict_copy['sample_count'] = total_sample_count
                else:
                    dict_copy['sample_count'] = 0
                
                enhanced_dictionaries.append(dict_copy)
            
            return enhanced_dictionaries
            
        except Exception as e:
            logging.exception(f"Error in get_dictionaries: {e}")
            return []
    
    def get_dictionary(self, dict_name):
        """
        Get a specific dictionary by name.
        
        Args:
            dict_name (str): Dictionary name
            
        Returns:
            dict: Dictionary information or None if not found
        """
        if not dict_name:
            logging.warning("No dictionary name provided to get_dictionary()")
            return None
            
        logging.info(f"Getting dictionary: {dict_name}")
        
        # Try to get dictionary directly by name
        dict_info = self.dictionaries.get('dictionaries', {}).get(dict_name)
        
        if not dict_info:
            # Try to find by case-insensitive name
            for name, dictionary in self.dictionaries.get('dictionaries', {}).items():
                if name.lower() == dict_name.lower() or dictionary.get('id', '').lower() == dict_name.lower():
                    dict_info = dictionary
                    break
                
        if dict_info:
            # Create a copy to avoid modifying the original
            dict_copy = copy.deepcopy(dict_info)
            
            # Ensure classes field exists
            if 'classes' not in dict_copy and 'sounds' in dict_copy:
                dict_copy['classes'] = dict_copy['sounds']
            elif 'classes' not in dict_copy:
                dict_copy['classes'] = []
                
            # Enhance with class details
            if dict_copy['classes']:
                class_details = []
                for class_name in dict_copy['classes']:
                    # Get class path
                    class_path = os.path.join(self.training_sounds_dir, class_name)
                    sample_count = 0
                    
                    # Count samples in the class directory
                    if os.path.exists(class_path) and os.path.isdir(class_path):
                        samples = [f for f in os.listdir(class_path) if f.endswith('.wav') or f.endswith('.mp3')]
                        sample_count = len(samples)
                        
                    class_details.append({
                        'name': class_name,
                        'sample_count': sample_count
                    })
                
                dict_copy['class_details'] = class_details
                
            return dict_copy
        
        return None
    
    def set_active_dictionary(self, dict_name):
        """
        Set the active dictionary.
        
        Args:
            dict_name (str): Dictionary name to set as active
            
        Returns:
            bool: True if successful, False otherwise
        """
        logging.info(f"Setting active dictionary: {dict_name}")
        
        if not dict_name:
            logging.error("Dictionary name is required")
            return False
            
        # Make sure dictionary exists
        if dict_name not in self.dictionaries.get('dictionaries', {}):
            logging.error(f"Dictionary '{dict_name}' not found")
            return False
            
        # Update active dictionary
        self.dictionaries['active_dictionary'] = dict_name
        
        # Save
        return self._save_dictionaries()
    
    def sync_dictionary_samples(self, dict_name):
        """
        Synchronize sample counts for a dictionary based on actual files.
        
        Args:
            dict_name (str): The name of the dictionary to sync
            
        Returns:
            dict: Result with success flag and message
        """
        try:
            logging.info(f"Syncing sample counts for dictionary '{dict_name}'")
            
            if not dict_name:
                logging.error("Dictionary name is required")
                return {
                    "success": False,
                    "error": "Dictionary name is required"
                }
                
            # Try to find dictionary by case-insensitive name
            dict_info = None
            for name, dictionary in self.dictionaries.get('dictionaries', {}).items():
                if name.lower() == dict_name.lower() or dictionary.get('id', '').lower() == dict_name.lower():
                    dict_info = dictionary
                    dict_name = name  # Update to exact case for later use
                    break
                    
            if not dict_info:
                logging.error(f"Dictionary '{dict_name}' not found")
                return {
                    "success": False,
                    "error": f"Dictionary '{dict_name}' not found"
                }
            
            # Ensure classes field exists
            if 'classes' not in dict_info and 'sounds' in dict_info:
                dict_info['classes'] = dict_info['sounds']
            elif 'classes' not in dict_info:
                dict_info['classes'] = []
                
            # Count samples for each class
            total_samples = 0
            class_details = []
            
            for class_name in dict_info['classes']:
                class_path = os.path.join(self.training_sounds_dir, class_name)
                sample_count = 0
                
                if os.path.exists(class_path) and os.path.isdir(class_path):
                    samples = [f for f in os.listdir(class_path) if f.endswith('.wav') or f.endswith('.mp3')]
                    sample_count = len(samples)
                    total_samples += sample_count
                    
                class_details.append({
                    'name': class_name,
                    'sample_count': sample_count
                })
                
            # Update dictionary with counts
            dict_info['class_details'] = class_details
            dict_info['sample_count'] = total_samples
            dict_info['updated_at'] = datetime.now().isoformat()
            
            # Save dictionaries
            self._save_dictionaries()
            
            logging.info(f"Successfully synced sample counts for dictionary '{dict_name}', found {total_samples} samples")
            return {
                "success": True, 
                "message": f"Successfully synced sample counts for dictionary '{dict_name}'",
                "sample_count": total_samples
            }
            
        except Exception as e:
            logging.exception(f"Error syncing sample counts for dictionary '{dict_name}': {e}")
            return {
                "success": False,
                "error": f"Error syncing sample counts: {str(e)}"
            }
    
    def add_class(self, dict_name, class_name):
        """
        Add a class to a dictionary.
        
        Args:
            dict_name (str): Dictionary name
            class_name (str): Class name
            
        Returns:
            dict: Result with success flag and message
        """
        try:
            logging.info(f"Adding class '{class_name}' to dictionary '{dict_name}'")
            
            if not dict_name or not class_name:
                logging.error("Dictionary name and class name are required")
                return {
                    "success": False,
                    "error": "Dictionary name and class name are required"
                }
            
            # Sanitize class name
            safe_class_name = class_name.replace(' ', '_').lower()
            
            # Find dictionary in the dictionaries
            dict_info = self.dictionaries["dictionaries"].get(dict_name)
            if not dict_info:
                # Try to find by case-insensitive name
                for name, dictionary in self.dictionaries.get('dictionaries', {}).items():
                    if name.lower() == dict_name.lower() or dictionary.get('id', '').lower() == dict_name.lower():
                        dict_info = dictionary
                        dict_name = name  # Update to exact case for later use
                        break
                    
            if not dict_info:
                logging.error(f"Dictionary '{dict_name}' not found")
                return {
                    "success": False,
                    "error": f"Dictionary '{dict_name}' not found"
                }
            
            # Ensure classes field exists
            if 'classes' not in dict_info:
                dict_info['classes'] = []
            
            # Check if class already exists
            if safe_class_name in dict_info['classes']:
                logging.warning(f"Class '{class_name}' already exists in dictionary '{dict_name}'")
                return {
                    "success": False,
                    "error": f"Class '{class_name}' already exists in this dictionary"
                }
            
            # Ensure the class directory exists in training sounds
            class_path = os.path.join(self.training_sounds_dir, safe_class_name)
            if not os.path.exists(class_path):
                try:
                    os.makedirs(class_path, exist_ok=True)
                    logging.info(f"Created class directory: {class_path}")
                except Exception as e:
                    logging.error(f"Failed to create class directory: {e}")
                    return {
                        "success": False,
                        "error": f"Failed to create class directory: {str(e)}"
                    }
            
            # Update dictionary with new class
            dict_info['classes'].append(safe_class_name)
            
            # Also update sounds field for backward compatibility
            if 'sounds' in dict_info:
                dict_info['sounds'] = dict_info['classes']
            
            dict_info['updated_at'] = datetime.now().isoformat()
            
            # Save dictionaries
            if self._save_dictionaries():
                logging.info(f"Added class '{class_name}' to dictionary '{dict_name}'")
                return {
                    "success": True,
                    "message": f"Added class '{class_name}' to dictionary '{dict_name}'"
                }
            else:
                logging.error(f"Failed to save dictionaries after adding class")
                return {
                    "success": False,
                    "error": "Failed to save dictionaries"
                }
        except Exception as e:
            logging.exception(f"Error adding class: {e}")
            return {
                "success": False,
                "error": str(e)
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
        dict_info = self.dictionaries["dictionaries"].get(safe_dict_name)
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
        # 1. Central sounds directory
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
        
        # Update dictionaries and sync sample counts
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
    
    def delete_dictionary(self, dict_name, delete_classes=False):
        """
        Delete a dictionary.
        
        Args:
            dict_name (str): Dictionary name
            delete_classes (bool, optional): Whether to delete class directories
            
        Returns:
            dict: Result with success flag and message
        """
        try:
            logging.info(f"Deleting dictionary: {dict_name}")
            
            if not dict_name:
                logging.error("Dictionary name is required")
                return {
                    "success": False,
                    "error": "Dictionary name is required"
                }
            
            # Find dictionary in the dictionaries
            dict_info = self.dictionaries["dictionaries"].get(dict_name)
            if not dict_info:
                # Try to find by case-insensitive name
                for name, dictionary in self.dictionaries.get('dictionaries', {}).items():
                    if name.lower() == dict_name.lower() or dictionary.get('id', '').lower() == dict_name.lower():
                        dict_info = dictionary
                        dict_name = name  # Update to exact case for later use
                        break
                    
            if not dict_info:
                logging.error(f"Dictionary '{dict_name}' not found")
                return {
                    "success": False,
                    "error": f"Dictionary '{dict_name}' not found"
                }
            
            # If this is the active dictionary, clear the active dictionary
            if self.dictionaries.get('active_dictionary') == dict_name:
                self.dictionaries['active_dictionary'] = None
            
            # Remove the dictionary
            self.dictionaries["dictionaries"].pop(dict_name, None)
            
            # Save dictionaries
            if not self._save_dictionaries():
                logging.error(f"Failed to save dictionaries after deleting '{dict_name}'")
                return {
                    "success": False,
                    "error": "Failed to save dictionaries"
                }
            
            # If requested, delete class directories (sound files)
            if delete_classes and 'classes' in dict_info:
                for class_name in dict_info['classes']:
                    try:
                        class_path = os.path.join(self.training_sounds_dir, class_name)
                        if os.path.exists(class_path):
                            # We don't actually delete the class directory, just report that we could
                            # This is a safeguard to prevent accidentally deleting sound files
                            logging.info(f"Would delete class directory: {class_path}")
                    except Exception as e:
                        logging.error(f"Error processing class directory {class_name}: {e}")
            
            logging.info(f"Dictionary '{dict_name}' deleted successfully")
            return {
                "success": True,
                "message": f"Dictionary '{dict_name}' deleted successfully"
            }
        except Exception as e:
            logging.exception(f"Error deleting dictionary: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def remove_class(self, dict_name, class_name):
        """
        Remove a class from a dictionary.
        
        Args:
            dict_name (str): Dictionary name
            class_name (str): Class name
            
        Returns:
            dict: Result with success flag and message
        """
        try:
            logging.info(f"Removing class '{class_name}' from dictionary '{dict_name}'")
            
            if not dict_name or not class_name:
                logging.error("Dictionary name and class name are required")
                return {
                    "success": False,
                    "error": "Dictionary name and class name are required"
                }
            
            # Sanitize class name
            safe_class_name = class_name.replace(' ', '_').lower()
            
            # Find dictionary in the dictionaries
            dict_info = self.dictionaries["dictionaries"].get(dict_name)
            if not dict_info:
                # Try to find by case-insensitive name
                for name, dictionary in self.dictionaries.get('dictionaries', {}).items():
                    if name.lower() == dict_name.lower() or dictionary.get('id', '').lower() == dict_name.lower():
                        dict_info = dictionary
                        dict_name = name  # Update to exact case for later use
                        break
                    
            if not dict_info:
                logging.error(f"Dictionary '{dict_name}' not found")
                return {
                    "success": False,
                    "error": f"Dictionary '{dict_name}' not found"
                }
            
            # Ensure classes field exists
            if 'classes' not in dict_info and 'sounds' in dict_info:
                dict_info['classes'] = dict_info['sounds']
            elif 'classes' not in dict_info:
                dict_info['classes'] = []
            
            # Check if class exists
            if safe_class_name not in dict_info['classes']:
                logging.warning(f"Class '{class_name}' not found in dictionary '{dict_name}'")
                return {
                    "success": False,
                    "error": f"Class '{class_name}' not found in dictionary '{dict_name}'"
                }
            
            # Remove class from dictionary
            dict_info['classes'].remove(safe_class_name)
            
            # Also update sounds field for backward compatibility
            if 'sounds' in dict_info:
                dict_info['sounds'] = dict_info['classes']
            
            dict_info['updated_at'] = datetime.now().isoformat()
            
            # Save dictionaries
            if self._save_dictionaries():
                logging.info(f"Removed class '{class_name}' from dictionary '{dict_name}'")
                return {
                    "success": True,
                    "message": f"Removed class '{class_name}' from dictionary '{dict_name}'"
                }
            else:
                logging.error(f"Failed to save dictionaries after removing class")
                return {
                    "success": False,
                    "error": "Failed to save dictionaries"
                }
        except Exception as e:
            logging.exception(f"Error removing class: {e}")
            return {
                "success": False,
                "error": str(e)
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
        if safe_dict_name not in self.dictionaries["dictionaries"]:
            return {
                "success": False,
                "error": f"Dictionary '{dict_name}' does not exist"
            }
        
        # Check if class exists in this dictionary
        dict_info = self.dictionaries["dictionaries"][safe_dict_name]
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

    # Add a delete_class method that calls remove_class for compatibility
    def delete_class(self, dict_name, class_name):
        """
        Delete a class from a dictionary.
        This is an alias for remove_class for API compatibility.
        
        Args:
            dict_name (str): Dictionary name
            class_name (str): Class name
            
        Returns:
            dict: Result with success flag and message
        """
        logging.info(f"delete_class called for '{class_name}' in '{dict_name}' (alias for remove_class)")
        return self.remove_class(dict_name, class_name)
