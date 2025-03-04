# SoundClassifier System

## Overview

SoundClassifier is a sophisticated application for training and using machine learning models to classify sounds. The system allows users to:

1. Create dictionaries of sounds
2. Record and manage sound samples
3. Train different types of machine learning models (CNN, RF, Ensemble)
4. Use trained models for real-time sound classification

## Architecture

The application follows a Flask-based web architecture with:

- **Backend**: Python-based API and processing engine
- **Frontend**: HTML templates with JavaScript for user interaction
- **Database**: File-based storage for sounds, models, and metadata

## Core Workflow

1. **Dictionary Creation**: Users define dictionaries containing sound classes
2. **Sound Recording**: Users record samples for each sound class
3. **Training**: ML models are trained on the sound samples
4. **Inference**: Trained models are used to classify new sounds in real-time

## File and Folder Structure

This section details how files are stored and retrieved throughout the application.

### Root Directory

- `config.py`: Core configuration settings
- `main.py`: Main application entry point
- `run.py`: Script to run the application

### Data Directory Structure

```
data/
├── dictionaries/
│   └── dictionaries.json        # Central registry of all dictionaries
├── models/
│   ├── models.json              # Central registry of all trained models
│   ├── best_cnn_model.h5        # Symlink to overall best model
│   ├── best_{dict_name}_model.h5 # Dictionary-specific best model symlinks
│   └── cnn/                     # CNN models directory
│       └── {dict_name}_cnn_{timestamp}/  # Model-specific folder
│           ├── {dict_name}_cnn_{timestamp}.h5         # Model file
│           └── {dict_name}_cnn_{timestamp}_metadata.json  # Metadata file
├── sounds/
│   ├── raw_sounds/              # Original sound recordings
│   ├── training_sounds/         # Processed sounds used for training
│   └── pending_verification_live_recording_sounds/  # Newly recorded sounds
└── analysis/                    # Model performance analysis
```

### Model Registry (models.json)

The `models.json` file is the central registry that tracks all trained models and their metadata. Its structure is:

```json
{
  "models": {
    "cnn": {
      "EhOh_cnn_20250304052326": {
        "id": "EhOh_cnn_20250304052326",
        "name": "EhOh CNN Model (20250304052326)",
        "type": "cnn",
        "dictionary": "EhOh",
        "file_path": "cnn/EhOh_cnn_20250304052326/EhOh_cnn_20250304052326.h5",
        "created_at": "2025-03-04T05:23:27.007081",
        "is_best": true,
        "class_names": ["eh", "oh"],
        "num_classes": 2,
        "input_shape": [163, 64, 1],
        "architecture": "...",
        "keras_version": "2.15.0",
        "tensorflow_version": "2.15.0"
      },
      // Additional models...
    },
    "rf": { /* RF models */ },
    "ensemble": { /* Ensemble models */ }
  },
  "counts": {
    "cnn": 27,
    "rf": 0,
    "ensemble": 0,
    "total": 27
  },
  "best_models": {
    "EhOh": "EhOh_cnn_20250304052326",
    "Default": "Default_cnn_20250303172659",
    // Other dictionaries...
  }
}
```

### Model Metadata Files

Each model has its own metadata file stored alongside the model file. The metadata contains:

- `class_names`: Array of sound classes the model was trained on
- `num_classes`: Number of sound classes
- `input_shape`: Shape of input data (important for CNN models)
- `architecture`: Model architecture details (for CNN models)
- Additional training statistics

## Critical Path for Model Training and Storage

1. **Model Training Process**:
   
   a. `src/services/training_service.py` handles model training
   
   ```python
   # In _train_cnn_model method:
   model_id = f"{dict_name.replace(' ', '_')}_cnn_{timestamp}"
   model_dir = os.path.join(self.config.BASE_DIR, 'data', 'models', 'cnn', model_id)
   os.makedirs(model_dir, exist_ok=True)
   model_file = os.path.join(model_dir, f"{model_id}.h5")
   model.save(model_file)
   
   # Create and save metadata
   metadata = {
       "class_names": class_names,
       "num_classes": num_classes,
       "input_shape": list(input_shape),
       "accuracy": float(metrics.get('accuracy', 0)),
       # Additional fields...
   }
   
   # Save metadata file
   metadata_file = save_model_metadata(model_dir, metadata)
   
   # Update models.json registry
   update_model_registry(model_id, 'cnn', dict_name, metadata, is_best)
   ```
   
   b. `src/ml/model_paths.py` handles the registry updating:
   
   ```python
   # Updates the models.json file with new model information
   def update_model_registry(model_id, model_type, dictionary_name, metadata, is_best):
       registry_path = os.path.join(Config.BASE_DIR, 'data', 'models', 'models.json')
       # Load existing registry or create new
       # Update registry entries
       # Save updated registry
   ```

2. **Best Model Tracking**:
   
   - Models can be marked as "best" for their dictionary
   - Symlinks are created in `data/models/` pointing to the best models
   - `best_{dict_name}_model.h5` links to the best model for a specific dictionary
   - `best_cnn_model.h5` links to the overall best model

## Critical Path for Model Loading and Inference

1. **Model Registry Access**:
   
   a. The `src/routes/ml_routes.py` file contains endpoints for model access:
   
   ```python
   @ml_bp.route('/api/ml/models')
   def get_available_models():
       # Load models.json
       # Return all available models with metadata
   ```
   
   b. The new model metadata endpoint:
   
   ```python
   @ml_bp.route('/model_metadata/<model_id>', methods=['GET'])
   def get_model_metadata_direct(model_id):
       # First try to get model from models.json registry
       # If not found or missing class_names, try loading from model's metadata file
       # Return model metadata including class_names
   ```

2. **Frontend Model Loading**:
   
   a. In `src/templates/predict.html`, models are loaded with:
   
   ```javascript
   function loadAvailableModels() {
       fetch('/api/ml/models')
           .then(response => response.json())
           .then(data => {
               // Process and display available models
           });
   }
   ```
   
   b. When a model is selected, the sound classes are loaded:
   
   ```javascript
   function loadSoundClassesForModel(modelId) {
       fetch(`/model_metadata/${encodeURIComponent(modelId)}`)
           .then(response => response.json())
           .then(data => {
               if (data.status === 'success' && data.metadata && data.metadata.class_names) {
                   updateSoundClasses(data.metadata.class_names);
               } else {
                   // Display error message
               }
           });
   }
   ```

3. **Inference Process**:
   
   a. The `src/services/inference_service.py` loads and uses models:
   
   ```python
   def load_model(self, model_id):
       # Determine model type
       # Load model file
       # Load metadata to get class_names
       # Return loaded model and metadata
   ```
   
   b. For predictions:
   
   ```python
   def predict(self, audio_data, model_id):
       model, metadata = self.load_model(model_id)
       # Process audio data
       # Make prediction
       # Return prediction with class_names from metadata
   ```

## Critical Issue: Sound Classes Not Displaying

The issue is that sound classes are not showing up in the web interface. Here's the flow that should happen:

1. User selects a model from the list on the predict page
2. Frontend calls `loadSoundClassesForModel(modelId)` which fetches from `/model_metadata/{modelId}`
3. This endpoint should return the model's metadata including `class_names`
4. The sound classes should be displayed in the UI

### Troubleshooting Steps

1. Check browser network tab:
   - Is the call to `/model_metadata/{modelId}` returning a 404 error?
   - What's the response content if it's not a 404?

2. Check server logs:
   - Are there any errors related to model loading or metadata retrieval?

3. Data validation:
   - Confirm that `class_names` exists in both the `models.json` entries and individual metadata files
   - Verify file paths are correct (use `.scripts/sync_models_registry.py --verbose` to check)

4. Endpoint verification:
   - Try accessing `/model_metadata/{modelId}` directly in the browser
   - Check if the endpoint is registered correctly in Flask

### Recent Fixes Attempted

1. Added a new direct endpoint for model metadata:
   ```python
   @ml_bp.route('/model_metadata/<model_id>', methods=['GET'])
   def get_model_metadata_direct(model_id):
       # Implementation...
   ```

2. Updated frontend code to use this endpoint:
   ```javascript
   fetch(`/model_metadata/${encodeURIComponent(modelId)}`)
   ```

3. Added enhanced error reporting in the frontend:
   ```javascript
   soundsList.innerHTML = `
       <div class="alert alert-warning">
           <div>
               <strong>Available Sound Classes for ${modelId}</strong><br>
               These are the sound classes that can be recognized.
           </div>
           <hr>
           <div>Error loading sound classes for ${modelId}: ${error.message}</div>
       </div>
   `;
   ```

4. Implemented a model registry synchronization tool:
   ```bash
   python .scripts/sync_models_registry.py --fix-best-model --verbose
   ```

## Setup and Run Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure the application:
   - Update `config.py` with your local paths
   - Ensure directories exist as specified

3. Run the application:
   ```bash
   python run.py
   ```

4. Access the web interface:
   - http://localhost:5002/

## Development Tools

- `sync_models_registry.py`: Updates and fixes the model registry
- Model browser in web interface: http://localhost:5002/predict
- Flask debug mode: `export FLASK_DEBUG=1` before running

## Contributing

When adding new models or modifying the system:

1. Follow the existing structure for model storage
2. Always update the models.json registry
3. Ensure metadata files contain required fields (especially class_names)
4. Test the entire flow from training to inference

## License

This project is proprietary and confidential. 