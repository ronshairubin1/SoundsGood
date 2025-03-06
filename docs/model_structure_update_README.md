# Model Structure Update Guide

## Background

The SoundClassifier application is further improving its model organization by implementing a more flexible model directory structure. Instead of storing models directly in the `data/models/cnn/` directory, each model will now have its own subdirectory to allow for additional model-specific files in the future.

## New Model Structure

The new model structure organizes models by type, with each model having its own directory:

```
data/models/
├── models.json                      # Index of all models
├── cnn/                             # CNN models directory
│   ├── EhOh_cnn_20250301113350/     # Individual model directory
│   │   ├── model.h5                 # Model file (standardized name)
│   │   ├── metadata.json            # Metadata file (standardized name)
│   │   └── ...                      # Other model-specific files (future)
│   ├── EhOh_cnn_20250301114545/     # Another model
│   │   ├── model.h5
│   │   ├── metadata.json
│   │   └── ...
│   └── ...
├── rf/                              # Random Forest models directory
│   └── ...
└── ensemble/                        # Ensemble models directory
    └── ...
```

Benefits of this structure:
1. Each model has its own isolated directory
2. Standardized filenames (`model.h5`, `metadata.json`) make code simpler
3. Room for additional model-specific files (e.g., training logs, performance metrics, sample data)
4. Cleaner organization when many models exist

## Migration Steps

To reorganize your models to the new structure, follow these steps:

1. **Run the Model Reorganization Script**:
   ```bash
   python reorganize_models.py
   ```
   
   This script will:
   - Create a subdirectory for each model
   - Move model files to their respective directories with standardized names
   - Update the models.json file with new file paths
   - Create a backup of the original models.json

2. **Test with the Updated Test Script**:
   ```bash
   python test_model_loading_updated.py
   ```
   
   This script will test loading models from the new structure to ensure everything works correctly.

3. **Update the Inference Service**:
   Replace `src/services/inference_service.py` with the updated version:
   ```bash
   mv src/services/inference_service_updated.py src/services/inference_service.py
   ```

4. **Remove Old Model Files** (Optional):
   After confirming everything works, you can remove the original model files that are now duplicated in the new structure. The reorganization script will prompt you to do this.

## Updated Components

The following components have been updated to work with the new model structure:

1. **reorganize_models.py** - Script to reorganize models into the new structure
2. **test_model_loading_updated.py** - Updated test script for the new structure
3. **src/services/inference_service_updated.py** - Updated inference service with backward compatibility

## Backward Compatibility

The updated code maintains backward compatibility in two ways:

1. The inference service can still load models from the old structure if needed
2. Models.json retains information about all models, regardless of structure

## Next Steps

After migration:

1. Use the new structure for all future model development
2. Consider adding additional model-specific files as needed
3. Update any code that directly references model file paths

## Additional Information

The standardized filenames within each model directory are:
- `model.h5` - The model file (for CNN models)
- `model.joblib` - The model file (for Random Forest models)
- `metadata.json` - The model metadata file

This standardization simplifies code as it no longer needs to construct complex filenames based on dictionary name, model type, and timestamp.

## Troubleshooting

If you encounter any issues:

1. Check that the models.json file has been properly updated
2. Verify that all model files were copied correctly to their new locations
3. Ensure the inference service is using the updated code
4. Check logs for specific error messages

For any persistent issues, please contact the development team. 