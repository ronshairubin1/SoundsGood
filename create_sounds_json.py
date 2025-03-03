import os
import json
import glob
import wave
import contextlib
from datetime import datetime

# Paths
DATA_DIR = "data"
SOUNDS_DIR = os.path.join(DATA_DIR, "sounds")
TRAINING_SOUNDS_DIR = os.path.join(SOUNDS_DIR, "training_sounds")
SOUNDS_JSON_PATH = os.path.join(SOUNDS_DIR, "sounds.json")
CLASSES_JSON_PATH = os.path.join(DATA_DIR, "classes", "classes.json")

# Ensure directories exist
os.makedirs(SOUNDS_DIR, exist_ok=True)

# Load classes information
classes_data = {}
try:
    with open(CLASSES_JSON_PATH, 'r') as f:
        classes_data = json.load(f)
except Exception as e:
    print(f"Error loading classes.json: {e}")
    classes_data = {"classes": {}}

# Function to get audio file duration
def get_wav_duration(file_path):
    try:
        with contextlib.closing(wave.open(file_path, 'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as e:
        print(f"Error getting duration for {file_path}: {e}")
        return 0

# Create sounds data
sounds = {}
total_sounds = 0

# Process each class
for class_name, class_info in classes_data.get("classes", {}).items():
    class_path = os.path.join(TRAINING_SOUNDS_DIR, class_name)
    
    # Skip if class directory doesn't exist
    if not os.path.isdir(class_path):
        continue
    
    # Get all .wav files for this class
    wav_files = glob.glob(os.path.join(class_path, "*.wav"))
    
    # Process each sound file
    for wav_file in wav_files:
        file_name = os.path.basename(wav_file)
        file_stats = os.stat(wav_file)
        
        # Get the file's creation or modification time
        file_time = datetime.fromtimestamp(file_stats.st_ctime).isoformat()
        
        # Get audio duration
        duration = get_wav_duration(wav_file)
        
        # Create sound entry
        sound_id = f"{class_name}_{os.path.splitext(file_name)[0]}"
        sounds[sound_id] = {
            "id": sound_id,
            "file_name": file_name,
            "class": class_name,
            "path": os.path.join(TRAINING_SOUNDS_DIR, class_name, file_name),
            "duration": duration,
            "size_bytes": file_stats.st_size,
            "created_at": file_time
        }
        
        total_sounds += 1

# Save the sounds.json file
with open(SOUNDS_JSON_PATH, 'w') as f:
    json.dump({"sounds": sounds, "total_count": total_sounds}, f, indent=4)

print(f"Created sounds.json with {total_sounds} sound files")
print(f"File saved to: {SOUNDS_JSON_PATH}") 