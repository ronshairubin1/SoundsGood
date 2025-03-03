# Sound Directory Structure

This directory contains all sound files used by the SoundClassifier application, organized by their purpose and stage in the processing pipeline.

## Directory Structure

- **raw_sounds/**: Original unprocessed recordings directly from the recording interface
  - Each sound class has its own subdirectory (`raw_sounds/{class}/`)
  - Files follow naming pattern: `{class_name}_raw_{timestamp}.wav`

- **pending_verification_live_recording_sounds/**: Processed sound chunks from live recordings awaiting verification
  - Each sound class has its own subdirectory
  - These are processed chunks ready for review before being added to the training set

- **uploaded_sounds/**: Temporary location for uploaded sound files during processing/prediction
  - Used for temporarily storing files uploaded through the API for prediction
  - Files are typically deleted after processing is complete
  - Files follow naming patterns like `rf_predict_{uuid}.wav`, `ensemble_predict_{uuid}.wav`, etc.

- **training_sounds/**: Verified sound files used for training models and for prediction
  - Each sound class has its own subdirectory (`training_sounds/{class}/`)
  - This is the primary source of sound data for training and prediction

- **test_sounds/**: Contains test recordings from development utilities
  - Used for microphone testing and other development purposes
  - Not used in the main application workflow

## Sound File Flow

1. **Recording**: New recordings are saved to `raw_sounds/{class}/`
2. **Processing & Chunking**: Raw recordings are processed and chunked into individual sounds in `pending_verification_live_recording_sounds/{class}/`
3. **Verification**: Verified chunks are moved to `training_sounds/{class}/` for use in training
4. **Prediction**: Files uploaded for prediction are temporarily stored in `uploaded_sounds/` during processing

## Legacy Paths

- The `pending_verification_live_recording_sounds/` directory was previously named `temp_sounds/` 