# Data Structure Migration Guide

This guide will walk you through the process of migrating your data from the root structure to the backend structure.

## Overview

The SoundClassifier application is transitioning from storing data in the root `/data/` directory to a more organized structure in `/backend/data/`. This migration will:

1. Move all sounds from `/data/sounds/` to `/backend/data/sounds/`
2. Move all feature cache files from `/feature_cache/` to `/backend/data/features/cache/`
3. Move all models from `/data/models/` to `/backend/data/models/`
4. Move all dictionaries from `/data/dictionaries/` to `/backend/data/dictionaries/`
5. Update configuration to use the new paths

## Migration Steps

Follow these steps in order to ensure a safe migration:

### Step 1: Backup Your Data

Before running any migration scripts, it's recommended to create a complete backup of your data:

```bash
# Create a backup directory
mkdir -p ~/SoundClassifier_backup

# Copy your data to the backup directory
cp -r data ~/SoundClassifier_backup/
cp -r feature_cache ~/SoundClassifier_backup/
cp config.py ~/SoundClassifier_backup/
```

### Step 2: Run the Migration Script

The migration script will copy all data to the new structure without deleting the original files:

```bash
python migrate_to_backend.py
```

This script will:
- Create all necessary directories in the backend structure
- Copy all data files to their new locations
- Update `config.py` to use the new paths

### Step 3: Update the Feature Extractor

Run the feature extractor update script to ensure it uses the new cache directory by default:

```bash
python update_feature_extractor.py
```

### Step 4: Validate the Migration

Run the validation script to ensure everything was migrated correctly:

```bash
python validate_migration.py
```

### Step 5: Test the Application

Test that the application still works with the new data structure:

```bash
python run.py
```

Verify that you can:
- Log in
- Record sounds
- Train models
- Make predictions

### Step 6: Move Original Data to Legacy (Optional)

If everything works correctly, you can run the migration script again and choose 'y' when prompted to move the original data to legacy folders:

```bash
python migrate_to_backend.py
```

## Troubleshooting

If you encounter any issues during the migration:

1. Check the log files:
   - `migration.log`
   - `feature_extractor_update.log`
   - `migration_validation.log`

2. Restore from your backup if needed.

3. If specific files are missing, you can re-run the migration script to copy them again.

4. If configuration paths are incorrect, check `config.py` and update paths manually if needed.

## After Migration

After successful migration:

1. All your data will be organized in the `/backend/data/` directory
2. The application will use a unified approach for feature extraction
3. Data processing for both training and inference will be consistent

Your data storage will now follow this structure:

```
backend/data/
├── sounds/
│   ├── raw/         (raw recordings)
│   ├── chopped/     (processed training sounds)
│   └── augmented/   (augmented variants of training sounds)
├── features/
│   ├── cache/       (extracted features cache)
│   ├── results/     (feature extraction results)
│   └── unified/     (unified feature representations)
├── models/
│   ├── cnn/         (CNN model files)
│   ├── rf/          (Random Forest model files)
│   └── ensemble/    (Ensemble model files)
└── dictionaries/    (sound dictionaries)
```

## Migration to Unified Model Architecture

### Completed Changes

1. **Blueprint Registration Issues**
   - Fixed duplicate endpoint functions in `ml_routes.py`
   - Renamed `view_analysis_details` to `view_analysis_detail` to avoid conflicts
   - Moved backup files to legacy directory

2. **Code Cleanup**
   - Removed unused imports
   - Fixed logging format strings
   - Improved error handling
   - Updated application context usage
   - Added docstrings to major route functions

3. **Feature Extractor Consolidation**
   - Updated `/predict_ensemble` route to use unified FeatureExtractor for both CNN and RF features
   - Updated `/predict_cnn` route to use unified FeatureExtractor
   - Moved legacy feature extractor backup files to `legacy/feature_extractors/`
   - Updated test files to use the unified FeatureExtractor

4. **Training Routes Consolidation**
   - Updated `/train_model` route to redirect to `/train_unified_model`
   - Ensured all model training uses the unified approach with consistent feature extraction
   - Added better progress reporting and error handling for training

5. **Application Context**
   - Replaced global variables with application context attributes
   - Updated prediction callbacks to use application context

### Issues Identified During Testing

1. **CNN Training Shape Mismatch**
   - The CNN training process encounters an error with feature shapes: "setting an array element with a sequence. The requested array has an inhomogeneous shape after 2 dimensions."
   - This occurs when trying to convert the list of CNN features to a numpy array
   - Likely caused by inconsistent feature shapes across different audio files

2. **RF Training Parameter Issue**
   - The RF training process has a parameter mismatch in the `_train_rf_model` method
   - The method is missing the `audio_dir` parameter or the calling code is incorrect
   - Need to update either the method signature or the calling code

3. **Feature Extraction Inconsistency**
   - Some routes still use different approaches for feature extraction
   - This can lead to inconsistencies between training and inference
   - Need to ensure all routes use the unified FeatureExtractor consistently

### Remaining Tasks

1. **Testing and Validation**
   - Test feature extraction with different audio files
   - Test training with different dictionaries
   - Test inference with different model types
   - Compare results between old and new approaches

2. **Documentation**
   - Update API documentation
   - Create developer guide for the unified model architecture
   - Document migration process for users

## Original Migration Notes

... (original content follows) 