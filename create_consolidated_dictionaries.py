import os
import json
import glob
from datetime import datetime

# Paths
DATA_DIR = "data"
DICTIONARIES_DIR = os.path.join(DATA_DIR, "dictionaries")
NEW_DICTIONARIES_JSON_PATH = os.path.join(DICTIONARIES_DIR, "dictionaries.json")
CONFIG_DICTIONARIES_PATH = "config/dictionaries.json"
ACTIVE_DICTIONARY_PATH = "config/active_dictionary.json"
LEGACY_METADATA_PATH = "legacy/dictionaries/metadata.json"
CLASSES_JSON_PATH = os.path.join(DATA_DIR, "classes", "classes.json")

# Ensure directories exist
os.makedirs(DICTIONARIES_DIR, exist_ok=True)

# Load existing dictionary information
dictionaries_data = {}
try:
    with open(CONFIG_DICTIONARIES_PATH, 'r') as f:
        dictionaries_data = json.load(f)
except Exception as e:
    print(f"Error loading dictionaries.json: {e}")
    dictionaries_data = {"dictionaries": []}

# Load active dictionary
active_dictionary = {}
try:
    with open(ACTIVE_DICTIONARY_PATH, 'r') as f:
        active_dictionary = json.load(f)
except Exception as e:
    print(f"Error loading active_dictionary.json: {e}")

# Load legacy metadata if available
legacy_metadata = {}
try:
    with open(LEGACY_METADATA_PATH, 'r') as f:
        legacy_metadata = json.load(f)
except Exception as e:
    print(f"Error loading legacy metadata.json: {e}")

# Load classes information
classes_data = {}
try:
    with open(CLASSES_JSON_PATH, 'r') as f:
        classes_data = json.load(f)
except Exception as e:
    print(f"Error loading classes.json: {e}")
    
# Create consolidated dictionaries data
consolidated_dictionaries = {}

# First, process dictionaries from config/dictionaries.json
for dictionary in dictionaries_data.get("dictionaries", []):
    dict_name = dictionary.get("name")
    if not dict_name:
        continue
        
    # Check if this dictionary exists in legacy metadata
    legacy_dict_info = None
    if legacy_metadata and "dictionaries" in legacy_metadata:
        # Find dictionary by name (case insensitive)
        for legacy_id, legacy_dict in legacy_metadata["dictionaries"].items():
            if legacy_dict.get("name", "").lower() == dict_name.lower():
                legacy_dict_info = legacy_dict
                break
    
    # Create consolidated dictionary entry
    consolidated_dictionaries[dict_name] = {
        "name": dict_name,
        "sounds": dictionary.get("sounds", []),
        "active": active_dictionary.get("name") == dict_name,
        "created_at": legacy_dict_info.get("created_at", datetime.now().isoformat()) if legacy_dict_info else datetime.now().isoformat(),
        "updated_at": legacy_dict_info.get("updated_at", datetime.now().isoformat()) if legacy_dict_info else datetime.now().isoformat(),
        "created_by": legacy_dict_info.get("created_by", "admin") if legacy_dict_info else "admin",
        "description": legacy_dict_info.get("description", "") if legacy_dict_info else "",
        "sample_count": sum([classes_data.get("classes", {}).get(class_name, {}).get("sample_count", 0) for class_name in dictionary.get("sounds", [])])
    }

# Save the consolidated dictionaries.json file
with open(NEW_DICTIONARIES_JSON_PATH, 'w') as f:
    json.dump({"dictionaries": consolidated_dictionaries, "active_dictionary": active_dictionary.get("name")}, f, indent=4)

print(f"Created consolidated dictionaries.json with {len(consolidated_dictionaries)} dictionaries")
print(f"File saved to: {NEW_DICTIONARIES_JSON_PATH}")
print(f"Active dictionary set to: {active_dictionary.get('name', 'None')}") 