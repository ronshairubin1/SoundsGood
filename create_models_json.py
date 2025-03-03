import os
import json
import glob
import re
from datetime import datetime

# Paths
DATA_DIR = "data"
MODELS_DIR = os.path.join(DATA_DIR, "models")
MODELS_CNN_DIR = os.path.join(MODELS_DIR, "cnn")
MODELS_RF_DIR = os.path.join(MODELS_DIR, "rf")
MODELS_ENSEMBLE_DIR = os.path.join(MODELS_DIR, "ensemble")
MODELS_JSON_PATH = os.path.join(MODELS_DIR, "models.json")
LEGACY_MODELS_DIR = "models"
DICTIONARIES_JSON_PATH = os.path.join(DATA_DIR, "dictionaries", "dictionaries.json")

# Ensure directories exist
os.makedirs(MODELS_CNN_DIR, exist_ok=True)
os.makedirs(MODELS_RF_DIR, exist_ok=True)
os.makedirs(MODELS_ENSEMBLE_DIR, exist_ok=True)

# Load dictionaries information
dictionaries_data = {}
try:
    with open(DICTIONARIES_JSON_PATH, 'r') as f:
        dictionaries_data = json.load(f)
except Exception as e:
    print(f"Error loading dictionaries.json: {e}")

# Create models data
models = {
    "cnn": {},
    "rf": {},
    "ensemble": {}
}

# Pattern to extract dictionary name and timestamp from model file name
# Example: "EhOh_cnn_20250301120736.h5" -> dictionary="EhOh", timestamp="20250301120736"
model_pattern = re.compile(r'([^_]+)_cnn_(\d+)\.h5')

# Get all model files
model_files = glob.glob(os.path.join(LEGACY_MODELS_DIR, "*.h5"))

for model_file in model_files:
    file_name = os.path.basename(model_file)
    match = model_pattern.match(file_name)
    
    if not match:
        continue
        
    dict_name, timestamp = match.groups()
    metadata_file = os.path.join(LEGACY_MODELS_DIR, f"{dict_name}_cnn_{timestamp}_metadata.json")
    
    # Get file stats
    file_stats = os.stat(model_file)
    file_time = datetime.fromtimestamp(file_stats.st_ctime).isoformat()
    
    # Load metadata if exists
    metadata = {}
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"Error loading metadata for {file_name}: {e}")
    
    # Create model entry
    model_id = f"{dict_name}_cnn_{timestamp}"
    models["cnn"][model_id] = {
        "id": model_id,
        "name": f"{dict_name} CNN Model ({timestamp})",
        "type": "cnn",
        "dictionary": dict_name,
        "created_at": file_time,
        "file_path": model_file,
        "size_bytes": file_stats.st_size,
        "class_names": metadata.get("class_names", []),
        "input_shape": metadata.get("input_shape", []),
        "num_classes": metadata.get("num_classes", 0)
    }

# Save the models.json file
with open(MODELS_JSON_PATH, 'w') as f:
    json.dump({
        "models": models,
        "counts": {
            "cnn": len(models["cnn"]),
            "rf": len(models["rf"]),
            "ensemble": len(models["ensemble"]),
            "total": len(models["cnn"]) + len(models["rf"]) + len(models["ensemble"])
        }
    }, f, indent=4)

print(f"Created models.json with {len(models['cnn'])} CNN models")
print(f"File saved to: {MODELS_JSON_PATH}")

# Create plan to move models to new structure
print("\nModels will need to be moved to the new directory structure:")
print(f"  - CNN models: {MODELS_CNN_DIR}")
print(f"  - RF models: {MODELS_RF_DIR}")
print(f"  - Ensemble models: {MODELS_ENSEMBLE_DIR}")
print("\nPlease run the following commands to move the models:")
print("----------------------------------------------------")
print(f"mkdir -p {MODELS_CNN_DIR}")

for model_id, model_info in models["cnn"].items():
    src_file = model_info["file_path"]
    dest_file = os.path.join(MODELS_CNN_DIR, os.path.basename(src_file))
    
    # Also copy metadata file if it exists
    metadata_src = os.path.splitext(src_file)[0] + "_metadata.json"
    metadata_dest = os.path.join(MODELS_CNN_DIR, os.path.basename(metadata_src))
    
    print(f"cp \"{src_file}\" \"{dest_file}\"")
    if os.path.exists(metadata_src):
        print(f"cp \"{metadata_src}\" \"{metadata_dest}\"") 