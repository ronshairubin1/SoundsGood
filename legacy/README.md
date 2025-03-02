# Legacy Code Directory

This directory contains legacy code from the SoundClassifier application that has been refactored into the new architecture. These files are kept for reference during the refactoring process to ensure no functionality is lost.

## Directory Structure

- `ml/`: Contains the original machine learning code, including model implementations, audio processing, and feature extraction
- `routes/`: Contains the original Flask route definitions
- `templates/`: Contains the original HTML templates
- `static/`: Contains original static assets (CSS, JS, images)

## Migration Process

These files are being gradually replaced by the new architecture:

1. Core components in `src/core/`
2. Service layer in `src/services/`
3. API routes in `src/api/`
4. New templates in `src/templates/`

## Usage

During the transition period, some of this code may still be referenced by the application. Once the refactoring is complete and all functionality has been verified, these files can be safely removed.

## File Mapping

Below is a mapping of legacy files to their new locations:

### ML Code
- `ml/audio_processing.py` → `src/core/audio/processor.py`
- `ml/cnn_classifier.py` → `src/core/models/cnn.py`
- `ml/rf_classifier.py` → `src/core/models/rf.py`
- `ml/ensemble_classifier.py` → `src/core/models/ensemble.py`
- `ml/feature_extractor.py` → Integrated into `src/core/audio/processor.py`
- `ml/trainer.py` → `src/services/training_service.py`
- `ml/inference.py` → `src/services/inference_service.py`

### Routes
- `routes/ml_routes.py` → Various API endpoints in `src/api/`

### Templates
- Various ML-related templates → New templates with improved UI/UX 