# Summary of .wav File Locations in SoundClassifier

This document provides a concise overview of where .wav files are sourced from and saved to throughout the SoundClassifier application.

## Main .wav File Locations

| Directory | Operation | Purpose |
|-----------|-----------|---------|
| `data/sounds/training_sounds/` | <--> | Main storage for verified sound files used for training models and prediction |
| `data/sounds/training_sounds/{class}/` | <--> | Class-specific directories (e.g., 'ah', 'eh') for verified training samples |
| `data/sounds/raw_sounds/` | --> | Storage for original unprocessed recordings before chunking |
| `data/sounds/raw_sounds/{class}/` | --> | Class-specific directories for raw recordings |
| `data/sounds/temp_sounds/` | --> | Temporary storage for sound chunks being processed |
| `data/sounds/temp_sounds/{class}/` | --> | Class-specific directories for temporary sound chunks |
| `root/temp/` | <--> | Temporary files used during ML operations (predictions, processing) |
| `root/` | --> | Root directory - used mainly for test recordings |

## Flow of .wav Files

The application handles .wav files in the following workflow:

1. **Recording & Initial Storage**:
   - New recordings are initially saved to `data/sounds/raw_sounds/{class}/` with naming pattern: `{class_name}_raw_{timestamp}.wav`

2. **Processing & Chunking**:
   - Recorded files are processed and chunked into individual sounds
   - Temporary chunks are stored in `data/sounds/temp_sounds/{class}/`

3. **Verification & Training**:
   - Verified chunks move to `data/sounds/training_sounds/{class}/`
   - These files are used for training models and for prediction

4. **Temporary ML Operations**:
   - During predictions and operations, temporary .wav files may be created in `root/temp/`
   - Test recordings may be saved to the root directory

## Key References in Code

- `Config.TRAINING_SOUNDS_DIR` points to `data/sounds/training_sounds/`
- `Config.RAW_SOUNDS_DIR` points to `data/sounds/raw_sounds/`
- `Config.PENDING_VERIFICATION_SOUNDS_DIR` points to `data/sounds/pending_verification_live_recording_sounds/`
- `Config.UPLOADED_SOUNDS_DIR` points to `data/sounds/uploaded_sounds/`
- `Config.TEST_SOUNDS_DIR` points to `data/sounds/test_sounds/`
- `Config.TEMP_DIR` points to `root/temp/`

## Legend
- `<--` files are read from this location
- `-->` files are written to this location
- `<-->` files are both read from and written to this location 