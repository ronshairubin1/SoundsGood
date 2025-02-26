# Legacy Source Files

This directory contains source files that have been replaced by newer implementations in the current architecture.

## Files

- `main_app.py`: An early, simplified version of the main application file (only 12 lines).
  - Replaced by the more comprehensive `main.py` in the root directory.
  - Added to legacy on Feb 26, 2024

- `test_mic.py`: A utility for testing microphone input.
  - This functionality is now integrated into the recording workflow in `static/js/recorder.js` and the API endpoints in `main.py`.
  - Added to legacy on Feb 26, 2024

## Usage

These files are kept for reference purposes during the ongoing refactoring of the application. If you need to recover specific functionality from these files, you can adapt them to work with the current architecture.

For the current implementation, please refer to `main.py` in the root directory and the audio recording functionality in `static/js/recorder.js`. 