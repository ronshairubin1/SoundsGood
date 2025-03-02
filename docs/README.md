# Sound Classifier Project
**Created: March 2, 2025 - 7:52 AM**

## Overview
This project is a complete sound classification system that records audio, processes it, trains machine learning models with data augmentation, and uses these models to classify sounds in real-time.

## Key Files and Their Purposes

### Core Application Files
1. **main.py** - The entry point for the Flask application
2. **run.py** - Script that manages the server startup process

### Backend Functionality

#### Training Pipeline
1. **src/core/train.py** - Handles the entire model training process:
   - Loads and processes audio data
   - Performs data augmentation
   - Creates train/test splits (80/20)
   - Trains models (CNN, RF, Ensemble)
   - Saves trained models as .h5 or .joblib files

2. **src/core/augmentation.py** - Implements data augmentation techniques:
   - Time stretching
   - Pitch shifting
   - Adding noise
   - Time masking
   - Frequency masking
   - Multiple augmentation combinations

3. **src/core/models/cnn_model.py** - CNN model architecture and training:
   - Model definition with convolutional layers
   - Training functions with callbacks
   - Evaluation metrics

4. **src/core/models/rf_model.py** - Random Forest implementation:
   - Feature extraction from audio
   - Model training and hyperparameter tuning
   - Prediction functions

5. **src/core/models/ensemble_model.py** - Ensemble model combining CNN and RF:
   - Weighted averaging of model outputs
   - Combined training approach

#### Model Management & API Routes
1. **src/ml_routes_fixed.py** - The primary file handling all ML-related endpoints:
   - Audio capture and processing
   - Model selection and loading
   - Real-time prediction streaming
   - Training initiation and progress tracking

2. **src/core/models/__init__.py** - Model factory and base classes:
   - Creates appropriate model types (CNN, RF, Ensemble)
   - Handles model loading and initialization

#### Audio Processing
1. **src/core/audio_processing.py** - Central audio processing utilities:
   - Feature extraction (mel spectrograms, MFCCs)
   - Sound detection algorithms
   - Preprocessing for different model types
   - Normalization and segmentation

2. **src/ml_routes_fixed.py** (SoundProcessor class) - Real-time processing:
   - Sound detection and boundary identification
   - Audio normalization and centering
   - Feature extraction tied to the model types

### Frontend Functionality
1. **src/templates/predict.html** - Prediction interface
2. **src/templates/train.html** - Training interface for:
   - Dataset selection and management
   - Model configuration
   - Training progress visualization
   - Augmentation settings

## Workflow Components

### 1. Class Creation
- Sound classes are defined when creating a new dataset
- Classes can be customized or selected from predefined dictionaries
- Class definitions feed into both training and prediction pipelines

### 2. Dictionary Creation from Classes
- Dictionaries map sound classes to specific model types
- Custom dictionaries can be created through the UI
- Metadata files store dictionary information alongside models

### 3. Recording and Preprocessing
- Audio capture pipeline for both training data collection and real-time prediction
- Specialized preprocessing based on model requirements
- Consistent preprocessing between training and inference

### 4. Data Augmentation
- **Critical component for improving model performance**
- Implemented in **src/core/augmentation.py**
- Techniques include:
  - Time stretching: Speeds up or slows down audio
  - Pitch shifting: Changes the pitch of the audio
  - Noise injection: Adds various types of noise
  - Time masking: Masks segments of the spectrogram in time
  - Frequency masking: Masks segments of the spectrogram in frequency
  - Combined augmentations: Multiple techniques applied sequentially
- Augmentation settings configurable through the UI
- Implementation leverages librosa for audio transformations

### 5. Training the Model (80/20 split)
- **Fully implemented in src/core/train.py**
- Training process:
  1. Loads audio samples for specified classes
  2. Applies configured augmentation techniques
  3. Splits data into training (80%) and validation (20%) sets
  4. Extracts features appropriate for the model type
  5. Initializes and trains the selected model architecture
  6. Evaluates on validation set
  7. Saves model files (.h5 for CNN, .joblib for RF)
  8. Creates metadata files with class information
- Supports CNN, RF, and Ensemble model types
- Training progress displayed in real-time via the UI

### 6. Using the Model for Predictions
- Real-time audio capture and processing
- Model selection from trained models
- Streaming predictions to the UI

### 7. Analyzing Predictions
- Performance tracking and metrics collection
- User feedback integration
- Model evaluation and comparison tools

## Recent Changes
- Added PyAudio support for audio capture
- Enhanced audio processing pipeline
- Implemented mock detector for testing
- Fixed API endpoints for model data

## What's Working
- The core application framework
- Mock detector for testing
- API endpoints for sound classes and statistics

## What's Left to Do
1. **Model Storage**:
   - Ensure model files are correctly stored and loaded
   - Verify metadata is properly associated with models

2. **Training Integration**:
   - Verify the training pipeline works end-to-end with the UI
   - Test augmentation settings and their effects

3. **Ensemble Models**:
   - Finalize ensemble model implementation
   - Test with combined CNN+RF approaches

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Ensure PyAudio is installed: `pip install pyaudio`
3. Run the application: `python run.py`
4. Access the web interface: http://localhost:5002

For training new models:
1. Navigate to the training interface
2. Select or create sound classes 
3. Configure augmentation settings
4. Start training
5. Monitor progress through the UI
6. Use trained models in the prediction interface

The application provides a complete pipeline from data collection through augmentation, training, and inference for sound classification tasks. 