import os
import json
import shutil

# Paths
DATA_DIR = "data"
MODELS_DIR = os.path.join(DATA_DIR, "models")
MODELS_CNN_DIR = os.path.join(MODELS_DIR, "cnn")
MODELS_RF_DIR = os.path.join(MODELS_DIR, "rf")
MODELS_ENSEMBLE_DIR = os.path.join(MODELS_DIR, "ensemble")
MODELS_JSON_PATH = os.path.join(MODELS_DIR, "models.json")
LEGACY_MODELS_DIR = "models"

# Ensure directories exist
os.makedirs(MODELS_CNN_DIR, exist_ok=True)
os.makedirs(MODELS_RF_DIR, exist_ok=True)
os.makedirs(MODELS_ENSEMBLE_DIR, exist_ok=True)

# Load models.json
models_data = {}
try:
    with open(MODELS_JSON_PATH, 'r') as f:
        models_data = json.load(f)
except Exception as e:
    print(f"Error loading models.json: {e}")
    exit(1)

# Move CNN models
print("Moving CNN models...")
for model_id, model_info in models_data.get("models", {}).get("cnn", {}).items():
    src_file = model_info["file_path"]
    dest_file = os.path.join(MODELS_CNN_DIR, os.path.basename(src_file))
    
    # Also copy metadata file if it exists
    metadata_src = os.path.splitext(src_file)[0] + "_metadata.json"
    metadata_dest = os.path.join(MODELS_CNN_DIR, os.path.basename(metadata_src))
    
    if os.path.exists(src_file):
        print(f"Copying {src_file} to {dest_file}")
        shutil.copy2(src_file, dest_file)
    else:
        print(f"Warning: Source file {src_file} not found")
        
    if os.path.exists(metadata_src):
        print(f"Copying {metadata_src} to {metadata_dest}")
        shutil.copy2(metadata_src, metadata_dest)
    else:
        print(f"Warning: Metadata file {metadata_src} not found")

# Update file paths in models.json
print("\nUpdating file paths in models.json...")
for model_type in ["cnn", "rf", "ensemble"]:
    for model_id, model_info in models_data.get("models", {}).get(model_type, {}).items():
        old_path = model_info["file_path"]
        if model_type == "cnn":
            new_path = os.path.join(MODELS_CNN_DIR, os.path.basename(old_path))
        elif model_type == "rf":
            new_path = os.path.join(MODELS_RF_DIR, os.path.basename(old_path))
        else:  # ensemble
            new_path = os.path.join(MODELS_ENSEMBLE_DIR, os.path.basename(old_path))
            
        models_data["models"][model_type][model_id]["file_path"] = new_path
        print(f"Updated path for {model_id}: {old_path} -> {new_path}")

# Save updated models.json
with open(MODELS_JSON_PATH, 'w') as f:
    json.dump(models_data, f, indent=4)
    
print("\nDone! Model files have been copied to the new directory structure.")
print("The models.json file has been updated with the new file paths.")
print("\nNOTE: The original model files in the 'models/' directory have NOT been deleted.")
print("You may want to delete them once you've verified the new structure works correctly.") 