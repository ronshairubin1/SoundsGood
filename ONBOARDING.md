# SoundsEasy Developer Onboarding

Welcome to the SoundsEasy project! This guide will help you get started quickly with the codebase.

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/SoundClassifier_v09.git
   cd SoundClassifier_v09
   ```

2. **Set up your environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python main.py
   ```

4. **Access the web interface**
   Open your browser and navigate to `http://localhost:5000`

## Project Structure At-a-Glance

```
SoundClassifier_v09/
├── backend/            # Core backend functionality 
├── src/                # Web application and services
│   ├── ml/             # Machine learning components
│   ├── services/       # Service layer
│   ├── templates/      # HTML templates
│   └── static/         # Static assets
├── tools/              # Utility scripts
├── legacy/             # Legacy code (for reference)
├── main.py             # Application entry point
└── config.py           # Configuration
```

## Key Components

### 1. Web Application (Flask)

The web interface is built with Flask and provides:
- User authentication
- Sound recording and verification
- Model training and inference
- Dictionary management

**Start Here:** `main.py` contains the main Flask application and routes.

### 2. Audio Processing Pipeline

The audio processing pipeline handles:
- Recording audio from the browser
- Segmenting audio into individual sounds
- Extracting features for machine learning
- Training and inference with ML models

**Start Here:** `src/services/recording_service.py` is the entry point for the new unified audio processing.

### 3. Feature Extraction

The feature extraction system:
- Extracts acoustic features from audio
- Implements caching for performance
- Provides consistent features across training and inference

**Start Here:** `backend/features/extractor.py` contains the unified feature extractor.

### 4. Machine Learning

The machine learning components include:
- Feature extraction and preprocessing
- Model training and cross-validation
- Inference and prediction

**Start Here:** `src/ml/inference.py` shows how the ML models are used for prediction.

## Common Tasks

### Adding a New Feature

1. Consider which layer the feature belongs in (backend, service, web interface)
2. Follow the existing patterns in similar files
3. Make sure to add tests for your new feature
4. Update documentation to reflect the new feature

### Fixing a Bug

1. Identify the affected component(s)
2. Add tests to reproduce the bug
3. Fix the bug
4. Run tests to ensure the fix works
5. Update documentation if necessary

### Modifying Audio Processing

1. Start with `src/services/recording_service.py`
2. Ensure any changes maintain compatibility with both recording and training
3. Test with real audio to verify the changes work as expected

### Working with the Feature Extractor

1. The `FeatureExtractor` class is the single source of truth for feature extraction
2. Check `backend/features/extractor.py` for the implementation
3. Use the caching functionality for performance
4. Ensure consistency between training and inference

## Best Practices

1. **Follow the Unified Approach**
   - Use the unified `FeatureExtractor` for all feature extraction
   - Use the `RecordingService` for audio processing
   - Follow the established directory structure for new code

2. **Maintain Backward Compatibility**
   - Support both unified and legacy approaches where needed
   - Test changes with both approaches
   - Document any breaking changes

3. **Write Tests**
   - Add tests for new functionality
   - Run existing tests before submitting changes
   - Verify audio processing changes with real audio

4. **Update Documentation**
   - Add docstrings to new functions and classes
   - Update README.md and other documentation when making significant changes
   - Keep API documentation up to date

## Recent Changes

The project is undergoing a transition to a more unified, maintainable architecture. Key recent changes include:

1. **Unified Feature Extraction**
   - Consolidated feature extraction into a single implementation
   - Added caching for performance improvement
   - Ensured consistency between training and inference

2. **RecordingService Implementation**
   - Unified audio processing through a service layer
   - Consistent processing across recording, verification, and training
   - Improved metadata handling

For more detailed information about these changes, see `RECENT_CHANGES.md`.

## Getting Help

If you have questions or need assistance:

1. Check the documentation in the codebase
2. Refer to `README.md` and `RECENT_CHANGES.md`
3. Look at existing code for examples
4. Reach out to the team for support

Happy coding! 