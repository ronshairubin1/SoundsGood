import os
import json
import glob
from datetime import datetime

# Paths
TRAINING_SOUNDS_DIR = "data/sounds/training_sounds"
CLASSES_DIR = "data/classes"
CLASSES_JSON_PATH = os.path.join(CLASSES_DIR, "classes.json")
CONFIG_DICTIONARIES_PATH = "config/dictionaries.json"
LEGACY_METADATA_PATH = "legacy/dictionaries/metadata.json"

# Ensure directories exist
os.makedirs(CLASSES_DIR, exist_ok=True)

# Get all class directories from training sounds
class_dirs = [d for d in os.listdir(TRAINING_SOUNDS_DIR) 
              if os.path.isdir(os.path.join(TRAINING_SOUNDS_DIR, d)) and not d.startswith('.')]

# Load dictionary information
dictionaries_data = {}
try:
    with open(CONFIG_DICTIONARIES_PATH, 'r') as f:
        dictionaries_data = json.load(f)
except Exception as e:
    print(f"Error loading dictionaries.json: {e}")

# Load legacy metadata if available
legacy_metadata = {}
try:
    with open(LEGACY_METADATA_PATH, 'r') as f:
        legacy_metadata = json.load(f)
except Exception as e:
    print(f"Error loading legacy metadata.json: {e}")

# Create class information
classes = {}

for class_name in class_dirs:
    # Skip non-class directories like 'training_sounds'
    if class_name == 'training_sounds':
        continue
        
    # Get sound samples for this class
    class_path = os.path.join(TRAINING_SOUNDS_DIR, class_name)
    samples = [os.path.basename(f) for f in glob.glob(os.path.join(class_path, "*.wav"))]
    
    # Create class entry
    classes[class_name] = {
        "name": class_name,
        "samples": samples,
        "sample_count": len(samples),
        "created_at": datetime.now().isoformat(),
        "in_dictionaries": []
    }
    
    # Add dictionary information (which dictionaries use this class)
    if dictionaries_data and "dictionaries" in dictionaries_data:
        for dictionary in dictionaries_data["dictionaries"]:
            if class_name in dictionary.get("sounds", []):
                classes[class_name]["in_dictionaries"].append(dictionary["name"])

# Save the classes.json file
with open(CLASSES_JSON_PATH, 'w') as f:
    json.dump({"classes": classes}, f, indent=4)

print(f"Created classes.json with {len(classes)} classes")
print(f"File saved to: {CLASSES_JSON_PATH}") 