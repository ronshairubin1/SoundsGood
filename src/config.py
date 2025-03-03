# File: src/config.py

import os
import json
import logging

class Config:
    # Get absolute paths
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
    STATIC_DIR = os.path.join(CURRENT_DIR, 'static')
    TEMP_DIR = os.path.join(STATIC_DIR, 'temp')
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
    TRAINING_SOUNDS_DIR = os.path.join(DATA_DIR, 'sounds', 'training_sounds')
    CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
    BASE_DIR = PROJECT_ROOT

    @classmethod
    def init_directories(cls):
        """Create all necessary directories"""
        for directory in [cls.STATIC_DIR, cls.TEMP_DIR, cls.TRAINING_SOUNDS_DIR, cls.CONFIG_DIR]:
            os.makedirs(directory, mode=0o755, exist_ok=True)
            logging.debug(f"Created directory: {directory}")

    @staticmethod
    def get_dictionary():
        """
        Get the active dictionary from the consolidated dictionaries.json file.
        """
        try:
            dictionaries_file = os.path.join(Config.BASE_DIR, 'data', 'dictionaries', 'dictionaries.json')
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
                    
                    return dictionary
                
                # If no match found, return the first dictionary or default
                if dictionaries and len(dictionaries) > 0:
                    first_dict = list(dictionaries.values())[0]
                    
                    # Ensure the sounds field exists (which might be called 'classes' in code)
                    if 'classes' not in first_dict and 'sounds' in first_dict:
                        first_dict['classes'] = first_dict['sounds']
                    
                    return first_dict
                
                return {
                    "name": "Default",
                    "classes": ["ah", "eh", "ee", "oh", "oo"]
                }
        except Exception as e:
            logging.error(f"Error loading dictionary: {e}")
            return {
                "name": "Default",
                "classes": ["ah", "eh", "ee", "oh", "oo"]
            }

    @classmethod
    def get_dictionaries(cls):
        """
        Get all dictionaries from the consolidated dictionaries.json file.
        """
        try:
            dictionaries_file = os.path.join(cls.BASE_DIR, 'data', 'dictionaries', 'dictionaries.json')
            with open(dictionaries_file, 'r') as f:
                data = json.load(f)
                dictionaries = data.get('dictionaries', {})
                
                # Convert from dictionary object to list for backwards compatibility
                dict_list = []
                for dict_name, dict_data in dictionaries.items():
                    # Ensure the sounds field exists (which might be called 'classes' in code)
                    if 'classes' not in dict_data and 'sounds' in dict_data:
                        dict_data['classes'] = dict_data['sounds']
                    
                    dict_list.append(dict_data)
                
                return dict_list
        except Exception as e:
            logging.error(f"Error loading dictionaries: {e}")
            return [{"name": "Default", "classes": ["ah", "eh", "ee", "oh", "oo"]}]

    @classmethod
    def save_dictionaries(cls, dictionaries, active_dictionary=None):
        """
        Save dictionaries to the consolidated dictionaries.json file.
        
        Args:
            dictionaries (list): List of dictionary objects
            active_dictionary (str, optional): Name of the active dictionary
        """
        try:
            dictionaries_file = os.path.join(cls.BASE_DIR, 'data', 'dictionaries', 'dictionaries.json')
            
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
                    
                # Handle the 'sounds' vs 'classes' field
                if 'sounds' not in dictionary and 'classes' in dictionary:
                    dictionary['sounds'] = dictionary['classes']
                
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
            dictionary (dict): Dictionary object to set as active
        """
        try:
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
