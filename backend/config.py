import json
import logging
import os

class Config:
    """
    Configuration settings for the application.
    Centralizes all configuration parameters for easy management.
    """
    # App directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Check and fix recursive backend/backend paths
    if os.path.basename(BASE_DIR) == 'backend' and os.path.basename(os.path.dirname(BASE_DIR)) == 'backend':
        # We're in a recursive backend/backend structure, move up one level
        BASE_DIR = os.path.dirname(BASE_DIR)
        logging.warning("Detected recursive backend/backend path. Fixing BASE_DIR to: %s", BASE_DIR)
    
    BACKEND_DATA_DIR = os.path.join(BASE_DIR, 'data')
    UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
    MODELS_DIR = os.path.join(BACKEND_DATA_DIR, 'models')
    TEMP_DIR = os.path.join(BACKEND_DATA_DIR, 'sounds', 'temp')
    DATA_DIR = BACKEND_DATA_DIR  # Point to backend data
    DICTIONARIES_DIR = os.path.join(BACKEND_DATA_DIR, 'dictionaries')
    ANALYSIS_DIR = os.path.join(BACKEND_DATA_DIR, 'analysis')
    LOGS_DIR = os.path.join(BASE_DIR, 'logs')
    
    # Define specific sound directories within backend/data/sounds
    RAW_SOUNDS_DIR = os.path.join(BACKEND_DATA_DIR, 'sounds', 'raw')
    PENDING_VERIFICATION_SOUNDS_DIR = os.path.join(BACKEND_DATA_DIR, 'sounds', 'pending')
    UPLOADED_SOUNDS_DIR = os.path.join(BACKEND_DATA_DIR, 'sounds', 'uploaded')
    TEST_SOUNDS_DIR = os.path.join(BACKEND_DATA_DIR, 'sounds', 'test')
    TRAINING_SOUNDS_DIR = os.path.join(BACKEND_DATA_DIR, 'sounds', 'training_sounds')
    AUGMENTED_SOUNDS_DIR = os.path.join(BACKEND_DATA_DIR, 'sounds', 'augmented')
    
    # Ensure all directories exist
    for directory in [UPLOAD_DIR, MODELS_DIR, TEMP_DIR, DATA_DIR, DICTIONARIES_DIR, ANALYSIS_DIR,
                      TRAINING_SOUNDS_DIR, RAW_SOUNDS_DIR, PENDING_VERIFICATION_SOUNDS_DIR,
                      UPLOADED_SOUNDS_DIR, TEST_SOUNDS_DIR, LOGS_DIR, AUGMENTED_SOUNDS_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # Audio processing settings
    SAMPLE_RATE = 16000
    SOUND_THRESHOLD = 0.02
    SILENCE_THRESHOLD = 0.02
    MIN_CHUNK_DURATION = 0.2
    MIN_SILENCE_DURATION = 0.1
    MAX_SILENCE_DURATION = 10.0
    
    # New audio processing parameters
    AMBIENT_NOISE_DURATION = 3.0
    SOUND_MULTIPLIER = 3.0
    SOUND_END_MULTIPLIER = 2.0
    PADDING_DURATION = 0.01
    ENABLE_STRETCHING = False
    TARGET_CHUNK_DURATION = 1.0
    AUTO_STOP_AFTER_SILENCE = 10.0
    
    # ML model settings
    CNN_PARAMS = {
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001,
        'n_mels': 128,
        'n_fft': 2048,
        'hop_length': 512
    }
    
    RF_PARAMS = {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1
    }
    
    ENSEMBLE_PARAMS = {
        'rf_weight': 0.5
    }
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev_key_for_testing')
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() in ['true', '1', 't']
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size
    
    # Logging settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    @classmethod
    def init_directories(cls):
        """
        Ensure all required directories exist.
        This is called during application startup.
        """
        directories = [
            cls.UPLOAD_DIR,
            cls.MODELS_DIR,
            cls.TEMP_DIR,
            cls.DATA_DIR,
            cls.DICTIONARIES_DIR,
            cls.ANALYSIS_DIR,
            cls.TRAINING_SOUNDS_DIR,
            cls.RAW_SOUNDS_DIR,
            cls.PENDING_VERIFICATION_SOUNDS_DIR,
            cls.UPLOADED_SOUNDS_DIR,
            cls.TEST_SOUNDS_DIR,
            cls.LOGS_DIR,
            cls.AUGMENTED_SOUNDS_DIR,
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
        return True
    
    @classmethod
    def get_dictionary(cls):
        """
        Get the active dictionary from the consolidated dictionaries.json file.
        
        Returns:
            dict: The active dictionary information
        """
        try:
            dictionaries_file = os.path.join(cls.DICTIONARIES_DIR, 'dictionaries.json')
            with open(dictionaries_file, 'r') as f:
                data = json.load(f)
                active_name = data.get('active_dictionary')
                
                # Find the dictionary with the matching name in the dictionaries object
                dictionaries = data.get('dictionaries', {})
                if active_name and active_name in dictionaries:
                    dictionary = dictionaries[active_name]
                    
                    # Ensure the sounds field exists (which might be called 'classes' in code)
                    if 'classes' not in dictionary and 'sounds' in dictionary:
                        dictionary['classes'] = dictionary['sounds']
                    elif 'sounds' not in dictionary and 'classes' in dictionary:
                        dictionary['sounds'] = dictionary['classes']
                    
                    return dictionary
                
                # If no match found, return the first dictionary or default
                if dictionaries and len(dictionaries) > 0:
                    first_dict = list(dictionaries.values())[0]
                    
                    # Ensure the sounds field exists (which might be called 'classes' in code)
                    if 'classes' not in first_dict and 'sounds' in first_dict:
                        first_dict['classes'] = first_dict['sounds']
                    elif 'sounds' not in first_dict and 'classes' in first_dict:
                        first_dict['sounds'] = first_dict['classes']
                    
                    return first_dict
                
                return {
                    "name": "Default",
                    "classes": ["ah", "eh", "ee", "oh", "oo"],
                    "sounds": ["ah", "eh", "ee", "oh", "oo"]
                }
        except Exception as e:
            logging.error(f"Error loading dictionary: {e}")
            return {
                "name": "Default",
                "classes": ["ah", "eh", "ee", "oh", "oo"],
                "sounds": ["ah", "eh", "ee", "oh", "oo"]
            }

    @classmethod
    def get_dictionaries(cls):
        """
        Get all dictionaries from the consolidated dictionaries.json file.
        
        Returns:
            list: List of dictionary objects
        """
        try:
            dictionaries_file = os.path.join(cls.DICTIONARIES_DIR, 'dictionaries.json')
            with open(dictionaries_file, 'r') as f:
                data = json.load(f)
                dictionaries = data.get('dictionaries', {})
                
                # Convert from dictionary object to list for backwards compatibility
                dict_list = []
                for dict_name, dict_data in dictionaries.items():
                    # Add dictionary name if missing
                    if 'name' not in dict_data:
                        dict_data['name'] = dict_name
                    
                    # Ensure the sounds field exists (which might be called 'classes' in code)
                    if 'classes' not in dict_data and 'sounds' in dict_data:
                        dict_data['classes'] = dict_data['sounds']
                    elif 'sounds' not in dict_data and 'classes' in dict_data:
                        dict_data['sounds'] = dict_data['classes']
                    
                    # Normalize sound/class names to lowercase for better matching
                    if 'sounds' in dict_data:
                        normalized_sounds = []
                        for sound in dict_data['sounds']:
                            # Handle both string and dictionary formats
                            if isinstance(sound, dict) and 'name' in sound:
                                normalized_sounds.append(sound['name'].lower())
                            elif isinstance(sound, str):
                                normalized_sounds.append(sound.lower())
                            else:
                                normalized_sounds.append(sound)
                        dict_data['normalized_sounds'] = normalized_sounds
                        
                        # Debug output
                        logging.debug(f"Dictionary '{dict_data.get('name')}' contains sounds: {dict_data['sounds']}")
                        logging.debug(f"Normalized sounds: {normalized_sounds}")
                    
                    dict_list.append(dict_data)
                
                return dict_list
        except Exception as e:
            logging.error(f"Error loading dictionaries: {e}")
            return [{"name": "Default", "classes": ["ah", "eh", "ee", "oh", "oo"], "sounds": ["ah", "eh", "ee", "oh", "oo"]}]

    @classmethod
    def save_dictionaries(cls, dictionaries, active_dictionary=None):
        """
        Save dictionaries to the consolidated dictionaries.json file.
        
        Args:
            dictionaries (list): List of dictionary objects
            active_dictionary (str, optional): Name of the active dictionary
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            dictionaries_file = os.path.join(cls.DICTIONARIES_DIR, 'dictionaries.json')
            
            # Read the current data first
            current_data = {}
            if os.path.exists(dictionaries_file):
                try:
                    with open(dictionaries_file, 'r') as f:
                        current_data = json.load(f)
                except Exception as e:
                    logging.error(f"Error reading existing dictionaries file: {e}")
            
            # If no active dictionary is specified, try to preserve the current one
            if active_dictionary is None:
                active_dictionary = current_data.get('active_dictionary')
            
            # Convert the list of dictionaries to a dictionary object with name as key
            dict_obj = {}
            for dictionary in dictionaries:
                # Ensure dictionary has a name
                name = dictionary.get('name')
                if not name:
                    continue
                    
                # Handle the 'sounds' vs 'classes' field for compatibility
                if 'sounds' not in dictionary and 'classes' in dictionary:
                    dictionary['sounds'] = dictionary['classes']
                elif 'classes' not in dictionary and 'sounds' in dictionary:
                    dictionary['classes'] = dictionary['sounds']
                
                # Add to dictionary object
                dict_obj[name] = dictionary
            
            # Prepare the data to save
            data_to_save = {
                'dictionaries': dict_obj,
                'active_dictionary': active_dictionary
            }
            
            # Write to file
            with open(dictionaries_file, 'w') as f:
                json.dump(data_to_save, f, indent=4)
                
            return True
        except Exception as e:
            logging.error(f"Error saving dictionaries: {e}")
            return False

    @classmethod
    def set_active_dictionary(cls, dictionary):
        """
        Set the active dictionary.
        
        Args:
            dictionary (dict or str): Dictionary object or name to set as active
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            dict_name = dictionary
            if isinstance(dictionary, dict):
                dict_name = dictionary.get('name')
                if not dict_name:
                    logging.error("Cannot set active dictionary: missing name")
                    return False
                
            # Save dictionaries with this one as active
            dictionaries = cls.get_dictionaries()
            return cls.save_dictionaries(dictionaries, dict_name)
        except Exception as e:
            logging.error(f"Error setting active dictionary: {e}")
            return False
    
    @classmethod
    def get_model_path(cls, model_name, model_type):
        """
        Get the full path for a model file based on its name and type.
        
        Args:
            model_name (str): Name of the model
            model_type (str): Type of model ('cnn', 'rf', or 'ensemble')
            
        Returns:
            str: Full path to the model file
        """
        extension = ''
        if model_type.lower() == 'cnn':
            extension = '.h5'
        elif model_type.lower() == 'rf':
            extension = '.joblib'
        
        return os.path.join(cls.MODELS_DIR, f"{model_name}{extension}")
    
    @classmethod
    def get_training_sounds_dir(cls):
        """
        Get the directory path for training sound files.
        
        Returns:
            str: Path to training sounds directory
        """
        return cls.TRAINING_SOUNDS_DIR
    
    @classmethod
    def get_upload_path(cls, filename):
        """
        Get the full path for an uploaded file.
        
        Args:
            filename (str): Name of the uploaded file
            
        Returns:
            str: Full path to the uploaded file
        """
        return os.path.join(cls.UPLOAD_DIR, filename)
    
    @classmethod
    def get_temp_path(cls, filename):
        """
        Get a temporary file path.
        
        Args:
            filename (str): Name for the temporary file
            
        Returns:
            str: Full path to the temporary file
        """
        # Returns the full path to a file in the temporary directory.
        return os.path.join(cls.TEMP_DIR, filename)