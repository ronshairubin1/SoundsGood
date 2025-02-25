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
    GOOD_SOUNDS_DIR = os.path.join(STATIC_DIR, 'goodsounds')
    CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')

    @classmethod
    def init_directories(cls):
        """Create all necessary directories"""
        for directory in [cls.STATIC_DIR, cls.TEMP_DIR, cls.GOOD_SOUNDS_DIR, cls.CONFIG_DIR]:
            os.makedirs(directory, mode=0o755, exist_ok=True)
            logging.debug(f"Created directory: {directory}")

    @staticmethod
    def get_dictionary():
        try:
            config_file = os.path.join(Config.CONFIG_DIR, 'active_dictionary.json')
            with open(config_file, 'r') as f:
                return json.load(f)
        except:
            return {
                "name": "Default",
                "sounds": ["ah", "eh", "ee", "oh", "oo"]
            }

    @classmethod
    def get_dictionaries(cls):
        try:
            with open(os.path.join(cls.CONFIG_DIR, 'dictionaries.json'), 'r') as f:
                return json.load(f)['dictionaries']
        except:
            return [{"name": "Default", "sounds": ["ah", "eh", "ee", "oh", "oo"]}]

    @classmethod
    def save_dictionaries(cls, dictionaries):
        with open(os.path.join(cls.CONFIG_DIR, 'dictionaries.json'), 'w') as f:
            json.dump({"dictionaries": dictionaries}, f, indent=4)

    @staticmethod
    def set_active_dictionary(new_dict):
        """
        Store the user-chosen dictionary as the active dictionary in memory or on disk.
        """
        # 1) Store in memory (if you have a class-level var for active dictionary):
        Config.ACTIVE_DICTIONARY = new_dict
        
        # 2) Optionally, save to disk so it persists between server restarts:
        active_path = os.path.join(Config.CONFIG_DIR, 'active_dictionary.json')
        with open(active_path, 'w') as f:
            json.dump(new_dict, f)
