# Tasks Remaining for Unified Model Transition

## Feature Extractor Consolidation

### Current Status
The codebase has many places where the unified `FeatureExtractor` from `backend.features.extractor` is being imported and used correctly. However, there are still some inconsistencies and legacy imports.

### Routes to Update

1. `/predict_rf` Route (ml_routes.py)
   - ✅ Currently uses the unified FeatureExtractor correctly
   - Ensure consistent parameters (sample_rate=16000)
   - Consider consolidating audio preprocessing to use SoundProcessor consistently

2. `/predict_ensemble` Route (ml_routes.py)
   - ✅ Uses unified FeatureExtractor for RF features
   - ❌ Still has separate preprocessing for CNN features
   - Should use the same preprocessing pipeline and feature extraction for both models

3. `/predict_cnn` Route (ml_routes.py)
   - ❌ Implementation needs to be checked and updated
   - Should use the unified FeatureExtractor for feature extraction
   - Should use consistent preprocessing pipeline

4. `/predict_sound_endpoint` Route (ml_routes.py)
   - ✅ Uses predict_sound function
   - Need to verify that predict_sound uses the unified FeatureExtractor

5. Sound Detector Classes
   - ✅ SoundDetector (inference.py) uses unified FeatureExtractor
   - ✅ SoundDetectorRF (sound_detector_rf.py) uses unified FeatureExtractor
   - ✅ SoundDetectorEnsemble (sound_detector_ensemble.py) uses unified FeatureExtractor
   - Verify parameters are consistent across all three implementations

### Legacy Code to Review

1. Test Modules
   - Update any test_feature_unification.py references to old extractors
   - Update test_forwarding.py to use unified FeatureExtractor directly

2. Backup Files
   - Move src/ml/feature_extractor.py.bak to legacy directory
   - Move src/ml/audio_processing.py.bak to legacy directory

## Model Training Consistency

1. `/train_model` Route
   - ❌ May still use legacy approach
   - Update to use the unified training workflow

2. `/train_unified_model` Route
   - ✅ Uses unified approach
   - Verify parameter consistency

3. Training Services
   - Ensure all training functions use consistent parameters and approaches
   - Consolidate duplicate code

## Inference Workflow

1. Prediction Routes
   - Ensure all prediction routes use same preprocessing steps
   - Consolidate duplicate preprocessing code
   - Ensure consistent model loading

2. Sound Detection
   - Ensure all sound detector implementations use consistent parameters
   - Consider consolidating into a single class with model type parameter

## Testing and Validation

1. Create Test Cases
   - Test feature extraction with different audio files
   - Test training with different dictionaries 
   - Test inference with different model types

2. Performance Validation
   - Compare results between old and new approaches
   - Document any differences in predictions 