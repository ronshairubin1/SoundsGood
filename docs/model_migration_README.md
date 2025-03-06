# Model Migration Guide

## Background

The SoundClassifier application currently stores trained models in two locations:
- `./models/` - The original location for model files
- `./data/models/` - The new consolidated location

This guide explains how to migrate your existing models to the new structure and ensure they work correctly.

## New Model Structure

The new model structure organizes models by type into a directory tree:

```
data/models/
├── models.json             # Index of all models
├── cnn/                    # CNN models directory
│   ├── EhOh_cnn_20250301113350.h5
│   ├── EhOh_cnn_20250301113350_metadata.json
│   └── ...
├── rf/                     # Random Forest models directory
│   └── ...
└── ensemble/               # Ensemble models directory
    └── ...
```

Each model has:
1. A model file (`.h5` for CNN models, `.pkl` for Random Forest models)
2. A metadata file (`.json`) containing class names, input shape, and other information

The `models.json` file serves as an index of all models, making it easier to find and load models.

## Migration Steps

The migration has been completed successfully! Here's a summary of what was done:

1. ✅ **Consolidate Models**: The `consolidate_models.py` script was run to copy CNN models and their metadata to the new location.
   - 19 CNN models were found and successfully migrated to the new structure.
   - All metadata files were properly handled.

2. ✅ **Update Configuration**: The `update_config.py` script was run to update the configuration to point to the new models directory.
   - A backup of the original configuration was created as `config.py.bak`.
   - The configuration now points to `data/models/` instead of `models/`.

3. ✅ **Test Model Loading**: The `test_model_loading.py` script was run to verify that models can be loaded from the new location.
   - All 19 CNN models were successfully loaded both directly and through the InferenceService.
   - No errors were encountered during model loading.

4. **Optional Cleanup** (Not Yet Done): 
   - Now that all models have been successfully migrated and tested, you can optionally remove the old model files from the `./models/` directory.
   - Before doing this, make sure you have a backup of your data.

## Important Files

The following files are important for model loading and saving:

- `src/core/models/cnn.py` - Contains the CNN model implementation
- `src/services/inference_service.py` - Handles loading models for inference
- `src/services/training_service.py` - Handles training and saving models
- `config.py` - Contains configuration settings including model directories

## Model Metadata

Each model has an associated metadata file that contains:
- Class names
- Input shape
- Number of classes
- Model architecture
- Creation date
- Training parameters
- Metrics history

This metadata is crucial for inference, as it provides the necessary information to preprocess input data and interpret model outputs.

## Next Steps

Now that the migration is complete, you can:

1. Continue using the application with the new model structure.
2. Train new models, which will automatically be saved to the new location.
3. Optionally clean up the old model files by running:
   ```
   python cleanup_old_models.py
   ```
   (Note: Create this script if you want to automate the cleanup process)

## Troubleshooting

If you encounter any issues:

1. Check that the model files exist in the expected location.
2. Verify that metadata files are present and contain valid JSON.
3. Ensure the configuration is pointing to the correct directory.
4. Check the logs for specific error messages.

For any persistent issues, please contact the development team. 