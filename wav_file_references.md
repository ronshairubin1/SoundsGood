# Analysis of .wav File References in Codebase

Generated on: 2025-03-02 13:52:32

This document contains an analysis of all references to `.wav` files in the codebase, excluding legacy code and sound data files.

## Table of Contents

1. [Summary](#summary)
2. [References by File](#references-by-file)
3. [References by Operation](#references-by-operation)

## Summary

Total .wav file references: 190

Operation types:

- See line content: 160
- File filtering: 26
- Path construction: 4

## References by File

### ./.scripts/create_favicon.py

| Line | Operation | Location | Content |
|------|-----------|----------|--------|
| ./.scripts/create_favicon.py | See line content | Create a simple favicon with a soundwave-like pattern using PIL | `Create a simple favicon with a soundwave-like pattern using PIL` |
| ./.scripts/create_favicon.py | See line content | # Draw a simple soundwave-like pattern | `# Draw a simple soundwave-like pattern` |
| ./.scripts/create_favicon.py | See line content | wave_color = (255, 255, 255)  # White | `wave_color = (255, 255, 255)  # White` |
| ./.scripts/create_favicon.py | See line content | fill=wave_color, | `fill=wave_color,` |
| ./.scripts/create_favicon.py | See line content | # Upper waves | `# Upper waves` |
| ./.scripts/create_favicon.py | See line content | fill=wave_color, | `fill=wave_color,` |
| ./.scripts/create_favicon.py | See line content | fill=wave_color, | `fill=wave_color,` |
| ./.scripts/create_favicon.py | See line content | # Lower waves | `# Lower waves` |
| ./.scripts/create_favicon.py | See line content | fill=wave_color, | `fill=wave_color,` |
| ./.scripts/create_favicon.py | See line content | fill=wave_color, | `fill=wave_color,` |

### ./docs/COMPREHENSIVE_PROJECT_HISTORY.md

| Line | Operation | Location | Content |
|------|-----------|----------|--------|
| ./docs/COMPREHENSIVE_PROJECT_HISTORY.md | See line content | - Added real-time waveform display | `- Added real-time waveform display` |
| ./docs/COMPREHENSIVE_PROJECT_HISTORY.md | See line content | - Enhanced waveform display | `- Enhanced waveform display` |

### ./main.py

| Line | Operation | Location | Content |
|------|-----------|----------|--------|
| ./main.py | See line content | 'icon': 'soundwave', | `'icon': 'soundwave',` |
| ./main.py | File filtering | Checking file extension | `samples = [f for f in os.listdir(class_path) if f.lower().endswith('.wav')]` |
| ./main.py | File filtering | Checking file extension | `samples = [f for f in os.listdir(class_path) if f.lower().endswith('.wav')]` |
| ./main.py | File filtering | Checking file extension | `if sample_file.lower().endswith('.wav'):` |
| ./main.py | File filtering | Checking file extension | `samples = [f for f in os.listdir(class_path) if f.lower().endswith('.wav')]` |
| ./main.py | File filtering | Checking file extension | `if sample_file.lower().endswith('.wav'):` |
| ./main.py | File filtering | Checking file extension | `if filename.endswith('.wav'):` |
| ./main.py | See line content | # The format should now be class_name/filename.wav | `# The format should now be class_name/filename.wav` |
| ./main.py | See line content | target_filename = f"{class_name}_{uuid.uuid4().hex[:8]}.wav" | `target_filename = f"{class_name}_{uuid.uuid4().hex[:8]}.wav"` |
| ./main.py | See line content | wav_filename = f"{sound_class}_raw_{timestamp}.wav" | `wav_filename = f"{sound_class}_raw_{timestamp}.wav"` |
| ./main.py | See line content | wav_path = os.path.join(raw_sounds_dir, wav_filename) | `wav_path = os.path.join(raw_sounds_dir, wav_filename)` |
| ./main.py | See line content | audio.export(wav_path, format="wav") | `audio.export(wav_path, format="wav")` |
| ./main.py | See line content | app.logger.info(f"Converted audio to WAV: {wav_path}") | `app.logger.info(f"Converted audio to WAV: {wav_path}")` |
| ./main.py | See line content | file_path=wav_path, | `file_path=wav_path,` |
| ./main.py | File filtering | Checking file extension | `if filename.endswith('.wav') and timestamp in filename:` |
| ./main.py | See line content | new_filename = f"{class_name}_{username}_{timestamp_now}_{uuid.uuid4().hex[:6]}.wav" | `new_filename = f"{class_name}_{username}_{timestamp_now}_{uuid.uuid4().hex[:6]}.wav"` |

### ./src/api/dashboard_api.py

| Line | Operation | Location | Content |
|------|-----------|----------|--------|
| ./src/api/dashboard_api.py | See line content | 'icon': 'soundwave', | `'icon': 'soundwave',` |

### ./src/audio_chunker.py

| Line | Operation | Location | Content |
|------|-----------|----------|--------|
| ./src/audio_chunker.py | See line content | from scipy.io import wavfile | `from scipy.io import wavfile` |
| ./src/audio_chunker.py | See line content | Splits a single wav file into smaller chunks based on silence detection, | `Splits a single wav file into smaller chunks based on silence detection,` |
| ./src/audio_chunker.py | See line content | rate, data = wavfile.read(filename) | `rate, data = wavfile.read(filename)` |
| ./src/audio_chunker.py | See line content | chunk_filename = filename.replace('.wav', f'_chunk_{i}.wav') | `chunk_filename = filename.replace('.wav', f'_chunk_{i}.wav')` |
| ./src/audio_chunker.py | See line content | wavfile.write(filename, rate, data_out.astype(np.int16)) | `wavfile.write(filename, rate, data_out.astype(np.int16))` |

### ./src/core/audio/processor.py

| Line | Operation | Location | Content |
|------|-----------|----------|--------|
| ./src/core/audio/processor.py | See line content | from scipy.io import wavfile | `from scipy.io import wavfile` |
| ./src/core/audio/processor.py | Path construction | Based on output_dir | `chunk_path = os.path.join(output_dir, f"{base_name}_chunk_{i+1}.wav")` |
| ./src/core/audio/processor.py | File filtering | Checking file extension | `wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]` |
| ./src/core/audio/processor.py | See line content | 'total_files': len(wav_files), | `'total_files': len(wav_files),` |
| ./src/core/audio/processor.py | See line content | for wav_file in wav_files: | `for wav_file in wav_files:` |
| ./src/core/audio/processor.py | See line content | file_path = os.path.join(class_path, wav_file) | `file_path = os.path.join(class_path, wav_file)` |

### ./src/ml/cnn_classifier.py

| Line | Operation | Location | Content |
|------|-----------|----------|--------|
| ./src/ml/cnn_classifier.py | See line content | # Gather .wav files | `# Gather .wav files` |
| ./src/ml/cnn_classifier.py | File filtering | Checking file extension | `files = [f for f in os.listdir(class_path) if f.endswith('.wav')]` |

### ./src/ml/feature_extractor.py

| Line | Operation | Location | Content |
|------|-----------|----------|--------|
| ./src/ml/feature_extractor.py | See line content | audio_path (str): Path to the .wav (or other) audio file. | `audio_path (str): Path to the .wav (or other) audio file.` |
| ./src/ml/feature_extractor.py | See line content | def save_wav(self, audio_data, sr, filename): | `def save_wav(self, audio_data, sr, filename):` |
| ./src/ml/feature_extractor.py | See line content | Save audio_data (numpy array) to a .wav file at the specified sample rate. | `Save audio_data (numpy array) to a .wav file at the specified sample rate.` |
| ./src/ml/feature_extractor.py | See line content | logging.error(f"Error saving wav file {filename}: {e}") | `logging.error(f"Error saving wav file {filename}: {e}")` |

### ./src/ml/inference.py

| Line | Operation | Location | Content |
|------|-----------|----------|--------|
| ./src/ml/inference.py | See line content | from scipy.io import wavfile | `from scipy.io import wavfile` |
| ./src/ml/inference.py | See line content | import wave | `import wave` |
| ./src/ml/inference.py | See line content | with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file: | `with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:` |
| ./src/ml/inference.py | See line content | with wave.open(temp_path, 'wb') as wf: | `with wave.open(temp_path, 'wb') as wf:` |
| ./src/ml/inference.py | See line content | wavfile.write("test_recording.wav", TEST_FS, test_recording_int) | `wavfile.write("test_recording.wav", TEST_FS, test_recording_int)` |

### ./src/ml/sound_detector_ensemble.py

| Line | Operation | Location | Content |
|------|-----------|----------|--------|
| ./src/ml/sound_detector_ensemble.py | See line content | import wave | `import wave` |
| ./src/ml/sound_detector_ensemble.py | See line content | with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file: | `with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:` |
| ./src/ml/sound_detector_ensemble.py | See line content | with wave.open(temp_path, 'wb') as wf: | `with wave.open(temp_path, 'wb') as wf:` |

### ./src/ml/sound_detector_rf.py

| Line | Operation | Location | Content |
|------|-----------|----------|--------|
| ./src/ml/sound_detector_rf.py | See line content | import wave | `import wave` |
| ./src/ml/sound_detector_rf.py | See line content | with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file: | `with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:` |
| ./src/ml/sound_detector_rf.py | See line content | with wave.open(temp_path, 'wb') as wf: | `with wave.open(temp_path, 'wb') as wf:` |

### ./src/ml/trainer.py

| Line | Operation | Location | Content |
|------|-----------|----------|--------|
| ./src/ml/trainer.py | See line content | Gather wavefiles from goodsounds_dir and extract "classical" features | `Gather wavefiles from goodsounds_dir and extract "classical" features` |
| ./src/ml/trainer.py | File filtering | Checking file extension | `files = [f for f in os.listdir(path_sound) if f.endswith('.wav')]` |
| ./src/ml/trainer.py | See line content | logging.info(f"# NEW LOG: Sound class='{sound}' has {len(files)} .wav files. path_sound={path_sound}") | `logging.info(f"# NEW LOG: Sound class='{sound}' has {len(files)} .wav files. path_sound={path_sound}")` |
| ./src/ml/trainer.py | See line content | audio = AudioSegment.from_file(filepath, format="wav") | `audio = AudioSegment.from_file(filepath, format="wav")` |
| ./src/ml/trainer.py | See line content | base_name = os.path.basename(filepath).replace('.wav', '_preprocessed.wav') | `base_name = os.path.basename(filepath).replace('.wav', '_preprocessed.wav')` |
| ./src/ml/trainer.py | See line content | audio.export(temp_path, format="wav") | `audio.export(temp_path, format="wav")` |

### ./src/ml_routes_fixed.py

| Line | Operation | Location | Content |
|------|-----------|----------|--------|
| ./src/ml_routes_fixed.py | See line content | import wave | `import wave` |
| ./src/ml_routes_fixed.py | See line content | with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file: | `with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:` |
| ./src/ml_routes_fixed.py | See line content | with wave.open(temp_path, 'wb') as wf: | `with wave.open(temp_path, 'wb') as wf:` |

### ./src/routes/ml_routes.py

| Line | Operation | Location | Content |
|------|-----------|----------|--------|
| ./src/routes/ml_routes.py | Path construction | Based on Config.TEMP_DIR | `temp_path = os.path.join(Config.TEMP_DIR, 'temp_for_rf.wav')` |
| ./src/routes/ml_routes.py | See line content | wavfile.write(temp_path, 16000, np.int16(trimmed_data * 32767)) | `wavfile.write(temp_path, 16000, np.int16(trimmed_data * 32767))` |
| ./src/routes/ml_routes.py | Path construction | Based on Config.TEMP_DIR | `temp_path = os.path.join(Config.TEMP_DIR, 'temp_ensemble.wav')` |
| ./src/routes/ml_routes.py | See line content | wav_data, _ = librosa.load(temp_path, sr=16000) | `wav_data, _ = librosa.load(temp_path, sr=16000)` |
| ./src/routes/ml_routes.py | See line content | cnn_features = sp.process_audio(wav_data) | `cnn_features = sp.process_audio(wav_data)` |
| ./src/routes/ml_routes.py | Path construction | Based on Config.TEMP_DIR | `temp_path = os.path.join(Config.TEMP_DIR, 'predict_temp.wav')` |
| ./src/routes/ml_routes.py | See line content | # Convert webm->wav with pydub | `# Convert webm->wav with pydub` |
| ./src/routes/ml_routes.py | See line content | wav_io = io.BytesIO() | `wav_io = io.BytesIO()` |
| ./src/routes/ml_routes.py | See line content | audio.export(wav_io, format="wav") | `audio.export(wav_io, format="wav")` |
| ./src/routes/ml_routes.py | See line content | wav_io.seek(0) | `wav_io.seek(0)` |
| ./src/routes/ml_routes.py | See line content | temp_filename = f"{sound}_{session['username']}_{timestamp}.wav" | `temp_filename = f"{sound}_{session['username']}_{timestamp}.wav"` |
| ./src/routes/ml_routes.py | See line content | f.write(wav_io.read()) | `f.write(wav_io.read())` |
| ./src/routes/ml_routes.py | See line content | new_filename = f"{sound}_{username}_{existing_count + 1}.wav" | `new_filename = f"{sound}_{username}_{existing_count + 1}.wav"` |
| ./src/routes/ml_routes.py | File filtering | Checking file extension | `if f.endswith('.wav') or f.endswith('.mp3')` |
| ./src/routes/ml_routes.py | File filtering | Checking file extension | `files = [f for f in os.listdir(sound_dir) if f.endswith('.wav')]` |
| ./src/routes/ml_routes.py | File filtering | Checking file extension | `sound_files = [f for f in os.listdir(sound_dir) if f.endswith('.wav')]` |
| ./src/routes/ml_routes.py | File filtering | Checking file extension | `files = [f for f in os.listdir(sound_dir) if f.endswith('.wav')]` |
| ./src/routes/ml_routes.py | File filtering | Checking file extension | `if file.filename.lower().endswith('.wav'):` |
| ./src/routes/ml_routes.py | See line content | f"{sound}_{session['username']}_{timestamp}_temp.wav") | `f"{sound}_{session['username']}_{timestamp}_temp.wav")` |

### ./src/services/dictionary_service.py

| Line | Operation | Location | Content |
|------|-----------|----------|--------|
| ./src/services/dictionary_service.py | File filtering | Checking file extension | `and f.lower().endswith(('.wav', '.mp3', '.ogg'))])` |
| ./src/services/dictionary_service.py | File filtering | Checking file extension | `and f.lower().endswith(('.wav', '.mp3', '.ogg'))])` |
| ./src/services/dictionary_service.py | File filtering | Checking file extension | `and f.lower().endswith(('.wav', '.mp3', '.ogg'))])` |
| ./src/services/dictionary_service.py | File filtering | Checking file extension | `and f.lower().endswith(('.wav', '.mp3', '.ogg'))])` |
| ./src/services/dictionary_service.py | See line content | # Ensure sample has .wav extension | `# Ensure sample has .wav extension` |
| ./src/services/dictionary_service.py | File filtering | Checking file extension | `if not sample_name.lower().endswith('.wav'):` |
| ./src/services/dictionary_service.py | See line content | sample_name += '.wav' | `sample_name += '.wav'` |
| ./src/services/dictionary_service.py | See line content | # Get all .wav files | `# Get all .wav files` |
| ./src/services/dictionary_service.py | File filtering | Checking file extension | `if filename.lower().endswith('.wav'):` |
| ./src/services/dictionary_service.py | File filtering | Checking file extension | `sample_files = [f for f in os.listdir(class_path) if f.lower().endswith('.wav')]` |
| ./src/services/dictionary_service.py | See line content | # Get all .wav files | `# Get all .wav files` |
| ./src/services/dictionary_service.py | File filtering | Checking file extension | `if filename.lower().endswith('.wav'):` |

### ./src/services/training_service.py

| Line | Operation | Location | Content |
|------|-----------|----------|--------|
| ./src/services/training_service.py | File filtering | Checking file extension | `wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]` |
| ./src/services/training_service.py | See line content | if not wav_files: | `if not wav_files:` |
| ./src/services/training_service.py | See line content | error_msg = f"No .wav files found in {class_path}" | `error_msg = f"No .wav files found in {class_path}"` |
| ./src/services/training_service.py | See line content | stats['original_counts'][class_dir] = len(wav_files) | `stats['original_counts'][class_dir] = len(wav_files)` |
| ./src/services/training_service.py | See line content | logging.info(f"Processing class {class_dir}: Found {len(wav_files)} audio files") | `logging.info(f"Processing class {class_dir}: Found {len(wav_files)} audio files")` |
| ./src/services/training_service.py | See line content | self.error_logs.append({'level': 'INFO', 'message': f"Processing class {class_dir}: Found {len(wav_files)} audio files"}) | `self.error_logs.append({'level': 'INFO', 'message': f"Processing class {class_dir}: Found {len(wav_files)} audio files"})` |
| ./src/services/training_service.py | See line content | for wav_file in wav_files: | `for wav_file in wav_files:` |
| ./src/services/training_service.py | See line content | file_path = os.path.join(class_path, wav_file) | `file_path = os.path.join(class_path, wav_file)` |
| ./src/services/training_service.py | File filtering | Checking file extension | `wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]` |
| ./src/services/training_service.py | See line content | if not wav_files: | `if not wav_files:` |
| ./src/services/training_service.py | See line content | logging.warning(f"No .wav files found in {class_path}") | `logging.warning(f"No .wav files found in {class_path}")` |
| ./src/services/training_service.py | See line content | stats['original_counts'][class_dir] = len(wav_files) | `stats['original_counts'][class_dir] = len(wav_files)` |
| ./src/services/training_service.py | See line content | for wav_file in wav_files: | `for wav_file in wav_files:` |
| ./src/services/training_service.py | See line content | file_path = os.path.join(class_path, wav_file) | `file_path = os.path.join(class_path, wav_file)` |
| ./src/services/training_service.py | File filtering | Checking file extension | `wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]` |
| ./src/services/training_service.py | See line content | if len(wav_files) < min_samples_per_class: | `if len(wav_files) < min_samples_per_class:` |
| ./src/services/training_service.py | See line content | insufficient_classes.append((class_dir, len(wav_files))) | `insufficient_classes.append((class_dir, len(wav_files)))` |

### ./src/templates/base.html

| Line | Operation | Location | Content |
|------|-----------|----------|--------|
| ./src/templates/base.html | See line content | <i class="bi bi-soundwave me-2"></i>SoundsEasy | `<i class="bi bi-soundwave me-2"></i>SoundsEasy` |
| ./src/templates/base.html | See line content | <a class="nav-link" href="/predict"><i class="bi bi-soundwave me-1"></i>Predict</a> | `<a class="nav-link" href="/predict"><i class="bi bi-soundwave me-1"></i>Predict</a>` |

### ./src/templates/dashboard.html

| Line | Operation | Location | Content |
|------|-----------|----------|--------|
| ./src/templates/dashboard.html | See line content | <i class="bi bi-soundwave"></i> | `<i class="bi bi-soundwave"></i>` |

### ./src/templates/dictionary_view.html

| Line | Operation | Location | Content |
|------|-----------|----------|--------|
| ./src/templates/dictionary_view.html | See line content | .waveform-container { | `.waveform-container {` |
| ./src/templates/dictionary_view.html | See line content | <div class="waveform-container" id="waveform"> | `<div class="waveform-container" id="waveform">` |
| ./src/templates/dictionary_view.html | See line content | <label for="audioFiles" class="form-label">Audio Files (.wav only)</label> | `<label for="audioFiles" class="form-label">Audio Files (.wav only)</label>` |
| ./src/templates/dictionary_view.html | See line content | <input type="file" class="form-control" id="audioFiles" accept=".wav" multiple required> | `<input type="file" class="form-control" id="audioFiles" accept=".wav" multiple required>` |
| ./src/templates/dictionary_view.html | See line content | <script src="https://unpkg.com/wavesurfer.js@6.6.3/dist/wavesurfer.js"></script> | `<script src="https://unpkg.com/wavesurfer.js@6.6.3/dist/wavesurfer.js"></script>` |
| ./src/templates/dictionary_view.html | See line content | let wavesurfer; | `let wavesurfer;` |
| ./src/templates/dictionary_view.html | See line content | // Initialize wavesurfer | `// Initialize wavesurfer` |
| ./src/templates/dictionary_view.html | See line content | wavesurfer = WaveSurfer.create({ | `wavesurfer = WaveSurfer.create({` |
| ./src/templates/dictionary_view.html | See line content | container: '#waveform', | `container: '#waveform',` |
| ./src/templates/dictionary_view.html | See line content | waveColor: 'rgba(67, 97, 238, 0.3)', | `waveColor: 'rgba(67, 97, 238, 0.3)',` |
| ./src/templates/dictionary_view.html | See line content | // Create audio blob and set up wavesurfer | `// Create audio blob and set up wavesurfer` |
| ./src/templates/dictionary_view.html | See line content | audioBlob = new Blob(recordedChunks, { type: 'audio/wav' }); | `audioBlob = new Blob(recordedChunks, { type: 'audio/wav' });` |
| ./src/templates/dictionary_view.html | See line content | wavesurfer.load(audioUrl); | `wavesurfer.load(audioUrl);` |
| ./src/templates/dictionary_view.html | See line content | wavesurfer.playPause(); | `wavesurfer.playPause();` |
| ./src/templates/dictionary_view.html | See line content | `recording_${new Date().toISOString().replace(/[:.-]/g, '_')}.wav`; | ``recording_${new Date().toISOString().replace(/[:.-]/g, '_')}.wav`;` |
| ./src/templates/dictionary_view.html | See line content | // Reset wavesurfer | `// Reset wavesurfer` |
| ./src/templates/dictionary_view.html | See line content | wavesurfer.empty(); | `wavesurfer.empty();` |

### ./src/templates/login.html

| Line | Operation | Location | Content |
|------|-----------|----------|--------|
| ./src/templates/login.html | See line content | <i class="bi bi-soundwave"></i> | `<i class="bi bi-soundwave"></i>` |

### ./src/templates/predict.html

| Line | Operation | Location | Content |
|------|-----------|----------|--------|
| ./src/templates/predict.html | See line content | .waveform-container { | `.waveform-container {` |
| ./src/templates/predict.html | See line content | <i class="bi bi-soundwave me-2"></i>Real-Time Predictor | `<i class="bi bi-soundwave me-2"></i>Real-Time Predictor` |
| ./src/templates/predict.html | See line content | <div class="waveform-container" id="waveform"> | `<div class="waveform-container" id="waveform">` |
| ./src/templates/predict.html | See line content | <h5><i class="bi bi-soundwave me-2"></i>Latest Prediction</h5> | `<h5><i class="bi bi-soundwave me-2"></i>Latest Prediction</h5>` |
| ./src/templates/predict.html | See line content | <script src="https://unpkg.com/wavesurfer.js@6.6.3/dist/wavesurfer.js"></script> | `<script src="https://unpkg.com/wavesurfer.js@6.6.3/dist/wavesurfer.js"></script>` |
| ./src/templates/predict.html | See line content | let wavesurfer; | `let wavesurfer;` |
| ./src/templates/predict.html | See line content | // Initialize wavesurfer | `// Initialize wavesurfer` |
| ./src/templates/predict.html | See line content | wavesurfer = WaveSurfer.create({ | `wavesurfer = WaveSurfer.create({` |
| ./src/templates/predict.html | See line content | container: '#waveform', | `container: '#waveform',` |
| ./src/templates/predict.html | See line content | waveColor: 'rgba(67, 97, 238, 0.3)', | `waveColor: 'rgba(67, 97, 238, 0.3)',` |

### ./src/templates/register.html

| Line | Operation | Location | Content |
|------|-----------|----------|--------|
| ./src/templates/register.html | See line content | <i class="bi bi-soundwave"></i> | `<i class="bi bi-soundwave"></i>` |

### ./src/templates/sound_class_view.html

| Line | Operation | Location | Content |
|------|-----------|----------|--------|
| ./src/templates/sound_class_view.html | See line content | .waveform-container { | `.waveform-container {` |
| ./src/templates/sound_class_view.html | See line content | <div class="waveform-container" id="waveform"> | `<div class="waveform-container" id="waveform">` |
| ./src/templates/sound_class_view.html | See line content | <script src="https://unpkg.com/wavesurfer.js@6.6.3/dist/wavesurfer.js"></script> | `<script src="https://unpkg.com/wavesurfer.js@6.6.3/dist/wavesurfer.js"></script>` |
| ./src/templates/sound_class_view.html | See line content | let wavesurfer; | `let wavesurfer;` |
| ./src/templates/sound_class_view.html | See line content | // Initialize wavesurfer and other functionality | `// Initialize wavesurfer and other functionality` |
| ./src/templates/sound_class_view.html | See line content | wavesurfer = WaveSurfer.create({ | `wavesurfer = WaveSurfer.create({` |
| ./src/templates/sound_class_view.html | See line content | container: '#waveform', | `container: '#waveform',` |
| ./src/templates/sound_class_view.html | See line content | waveColor: 'rgba(67, 97, 238, 0.3)', | `waveColor: 'rgba(67, 97, 238, 0.3)',` |
| ./src/templates/sound_class_view.html | See line content | // Create audio blob and set up wavesurfer | `// Create audio blob and set up wavesurfer` |
| ./src/templates/sound_class_view.html | See line content | audioBlob = new Blob(recordedChunks, { type: 'audio/wav' }); | `audioBlob = new Blob(recordedChunks, { type: 'audio/wav' });` |
| ./src/templates/sound_class_view.html | See line content | wavesurfer.load(audioUrl); | `wavesurfer.load(audioUrl);` |
| ./src/templates/sound_class_view.html | See line content | wavesurfer.playPause(); | `wavesurfer.playPause();` |
| ./src/templates/sound_class_view.html | See line content | `recording_${new Date().toISOString().replace(/[:.-]/g, '_')}.wav`; | ``recording_${new Date().toISOString().replace(/[:.-]/g, '_')}.wav`;` |
| ./src/templates/sound_class_view.html | See line content | // Reset wavesurfer | `// Reset wavesurfer` |
| ./src/templates/sound_class_view.html | See line content | wavesurfer.empty(); | `wavesurfer.empty();` |

### ./src/templates/sounds_management.html

| Line | Operation | Location | Content |
|------|-----------|----------|--------|
| ./src/templates/sounds_management.html | See line content | <div class="waveform-container" id="waveform"> | `<div class="waveform-container" id="waveform">` |
| ./src/templates/sounds_management.html | See line content | <script src="https://unpkg.com/wavesurfer.js@6.6.3/dist/wavesurfer.js"></script> | `<script src="https://unpkg.com/wavesurfer.js@6.6.3/dist/wavesurfer.js"></script>` |
| ./src/templates/sounds_management.html | See line content | let wavesurfer; | `let wavesurfer;` |
| ./src/templates/sounds_management.html | See line content | // Initialize wavesurfer | `// Initialize wavesurfer` |
| ./src/templates/sounds_management.html | See line content | wavesurfer = WaveSurfer.create({ | `wavesurfer = WaveSurfer.create({` |
| ./src/templates/sounds_management.html | See line content | container: '#waveform', | `container: '#waveform',` |
| ./src/templates/sounds_management.html | See line content | waveColor: 'rgba(67, 97, 238, 0.3)', | `waveColor: 'rgba(67, 97, 238, 0.3)',` |
| ./src/templates/sounds_management.html | See line content | // Create audio blob and set up wavesurfer | `// Create audio blob and set up wavesurfer` |
| ./src/templates/sounds_management.html | See line content | audioBlob = new Blob(recordedChunks, { type: 'audio/wav' }); | `audioBlob = new Blob(recordedChunks, { type: 'audio/wav' });` |
| ./src/templates/sounds_management.html | See line content | wavesurfer.load(audioUrl); | `wavesurfer.load(audioUrl);` |
| ./src/templates/sounds_management.html | See line content | wavesurfer.playPause(); | `wavesurfer.playPause();` |
| ./src/templates/sounds_management.html | See line content | `recording_${new Date().toISOString().replace(/[:.-]/g, '_')}.wav`; | ``recording_${new Date().toISOString().replace(/[:.-]/g, '_')}.wav`;` |
| ./src/templates/sounds_management.html | See line content | // Reset wavesurfer | `// Reset wavesurfer` |
| ./src/templates/sounds_management.html | See line content | wavesurfer.empty(); | `wavesurfer.empty();` |

### ./src/templates/sounds_record.html

| Line | Operation | Location | Content |
|------|-----------|----------|--------|
| ./src/templates/sounds_record.html | See line content | .waveform-container { | `.waveform-container {` |
| ./src/templates/sounds_record.html | See line content | <i class="bi bi-soundwave me-2"></i>Step 2: Record Sound | `<i class="bi bi-soundwave me-2"></i>Step 2: Record Sound` |
| ./src/templates/sounds_record.html | See line content | <i class="bi bi-soundwave me-1"></i>Check Ambient Noise | `<i class="bi bi-soundwave me-1"></i>Check Ambient Noise` |
| ./src/templates/sounds_record.html | See line content | <div id="waveform" class="my-4"></div> | `<div id="waveform" class="my-4"></div>` |
| ./src/templates/sounds_record.html | See line content | formData.append('audio', blob, 'recording.wav'); | `formData.append('audio', blob, 'recording.wav');` |

### ./src/templates/upload_sounds.html

| Line | Operation | Location | Content |
|------|-----------|----------|--------|
| ./src/templates/upload_sounds.html | See line content | 2. Choose one or more .wav files to upload<br> | `2. Choose one or more .wav files to upload<br>` |
| ./src/templates/upload_sounds.html | See line content | accept=".wav" | `accept=".wav"` |
| ./src/templates/upload_sounds.html | See line content | You can select multiple .wav files at once | `You can select multiple .wav files at once` |

### ./src/templates/verify.html

| Line | Operation | Location | Content |
|------|-----------|----------|--------|
| ./src/templates/verify.html | See line content | <i class="bi bi-soundwave me-2"></i> | `<i class="bi bi-soundwave me-2"></i>` |

### ./src/test_mic.py

| Line | Operation | Location | Content |
|------|-----------|----------|--------|
| ./src/test_mic.py | See line content | from scipy.io import wavfile | `from scipy.io import wavfile` |
| ./src/test_mic.py | See line content | wavfile.write("test_recording.wav", fs, recording_int) | `wavfile.write("test_recording.wav", fs, recording_int)` |

### ./static/js/recorder.js

| Line | Operation | Location | Content |
|------|-----------|----------|--------|
| ./static/js/recorder.js | See line content | const blob = new Blob(this.audioChunks, { type: 'audio/wav' }); | `const blob = new Blob(this.audioChunks, { type: 'audio/wav' });` |

### ./wav_file_references.md

| Line | Operation | Location | Content |
|------|-----------|----------|--------|
| ./wav_file_references.md | See line content | # Analysis of .wav File References in Codebase | `# Analysis of .wav File References in Codebase` |
| ./wav_file_references.md | See line content | This document contains an analysis of all references to `.wav` files in the codebase, excluding legacy code and sound data files. | `This document contains an analysis of all references to `.wav` files in the codebase, excluding legacy code and sound data files.` |
| ./wav_file_references.md | See line content | Total .wav file references: 0 | `Total .wav file references: 0` |

## References by Operation

### File filtering

| File | Location | Content |
|------|----------|--------|
| ./main.py | Checking file extension | `samples = [f for f in os.listdir(class_path) if f.lower().endswith('.wav')]` |
| ./main.py | Checking file extension | `samples = [f for f in os.listdir(class_path) if f.lower().endswith('.wav')]` |
| ./main.py | Checking file extension | `if sample_file.lower().endswith('.wav'):` |
| ./main.py | Checking file extension | `samples = [f for f in os.listdir(class_path) if f.lower().endswith('.wav')]` |
| ./main.py | Checking file extension | `if sample_file.lower().endswith('.wav'):` |
| ./main.py | Checking file extension | `if filename.endswith('.wav'):` |
| ./main.py | Checking file extension | `if filename.endswith('.wav') and timestamp in filename:` |
| ./src/core/audio/processor.py | Checking file extension | `wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]` |
| ./src/ml/cnn_classifier.py | Checking file extension | `files = [f for f in os.listdir(class_path) if f.endswith('.wav')]` |
| ./src/ml/trainer.py | Checking file extension | `files = [f for f in os.listdir(path_sound) if f.endswith('.wav')]` |
| ./src/routes/ml_routes.py | Checking file extension | `if f.endswith('.wav') or f.endswith('.mp3')` |
| ./src/routes/ml_routes.py | Checking file extension | `files = [f for f in os.listdir(sound_dir) if f.endswith('.wav')]` |
| ./src/routes/ml_routes.py | Checking file extension | `sound_files = [f for f in os.listdir(sound_dir) if f.endswith('.wav')]` |
| ./src/routes/ml_routes.py | Checking file extension | `files = [f for f in os.listdir(sound_dir) if f.endswith('.wav')]` |
| ./src/routes/ml_routes.py | Checking file extension | `if file.filename.lower().endswith('.wav'):` |
| ./src/services/dictionary_service.py | Checking file extension | `and f.lower().endswith(('.wav', '.mp3', '.ogg'))])` |
| ./src/services/dictionary_service.py | Checking file extension | `and f.lower().endswith(('.wav', '.mp3', '.ogg'))])` |
| ./src/services/dictionary_service.py | Checking file extension | `and f.lower().endswith(('.wav', '.mp3', '.ogg'))])` |
| ./src/services/dictionary_service.py | Checking file extension | `and f.lower().endswith(('.wav', '.mp3', '.ogg'))])` |
| ./src/services/dictionary_service.py | Checking file extension | `if not sample_name.lower().endswith('.wav'):` |
| ./src/services/dictionary_service.py | Checking file extension | `if filename.lower().endswith('.wav'):` |
| ./src/services/dictionary_service.py | Checking file extension | `sample_files = [f for f in os.listdir(class_path) if f.lower().endswith('.wav')]` |
| ./src/services/dictionary_service.py | Checking file extension | `if filename.lower().endswith('.wav'):` |
| ./src/services/training_service.py | Checking file extension | `wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]` |
| ./src/services/training_service.py | Checking file extension | `wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]` |
| ./src/services/training_service.py | Checking file extension | `wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]` |

### Path construction

| File | Location | Content |
|------|----------|--------|
| ./src/core/audio/processor.py | Based on output_dir | `chunk_path = os.path.join(output_dir, f"{base_name}_chunk_{i+1}.wav")` |
| ./src/routes/ml_routes.py | Based on Config.TEMP_DIR | `temp_path = os.path.join(Config.TEMP_DIR, 'temp_for_rf.wav')` |
| ./src/routes/ml_routes.py | Based on Config.TEMP_DIR | `temp_path = os.path.join(Config.TEMP_DIR, 'temp_ensemble.wav')` |
| ./src/routes/ml_routes.py | Based on Config.TEMP_DIR | `temp_path = os.path.join(Config.TEMP_DIR, 'predict_temp.wav')` |

### See line content

| File | Location | Content |
|------|----------|--------|
| ./docs/COMPREHENSIVE_PROJECT_HISTORY.md | - Added real-time waveform display | `- Added real-time waveform display` |
| ./docs/COMPREHENSIVE_PROJECT_HISTORY.md | - Enhanced waveform display | `- Enhanced waveform display` |
| ./.scripts/create_favicon.py | Create a simple favicon with a soundwave-like pattern using PIL | `Create a simple favicon with a soundwave-like pattern using PIL` |
| ./.scripts/create_favicon.py | # Draw a simple soundwave-like pattern | `# Draw a simple soundwave-like pattern` |
| ./.scripts/create_favicon.py | wave_color = (255, 255, 255)  # White | `wave_color = (255, 255, 255)  # White` |
| ./.scripts/create_favicon.py | fill=wave_color, | `fill=wave_color,` |
| ./.scripts/create_favicon.py | # Upper waves | `# Upper waves` |
| ./.scripts/create_favicon.py | fill=wave_color, | `fill=wave_color,` |
| ./.scripts/create_favicon.py | fill=wave_color, | `fill=wave_color,` |
| ./.scripts/create_favicon.py | # Lower waves | `# Lower waves` |
| ./.scripts/create_favicon.py | fill=wave_color, | `fill=wave_color,` |
| ./.scripts/create_favicon.py | fill=wave_color, | `fill=wave_color,` |
| ./static/js/recorder.js | const blob = new Blob(this.audioChunks, { type: 'audio/wav' }); | `const blob = new Blob(this.audioChunks, { type: 'audio/wav' });` |
| ./wav_file_references.md | # Analysis of .wav File References in Codebase | `# Analysis of .wav File References in Codebase` |
| ./wav_file_references.md | This document contains an analysis of all references to `.wav` files in the codebase, excluding legacy code and sound data files. | `This document contains an analysis of all references to `.wav` files in the codebase, excluding legacy code and sound data files.` |
| ./wav_file_references.md | Total .wav file references: 0 | `Total .wav file references: 0` |
| ./main.py | 'icon': 'soundwave', | `'icon': 'soundwave',` |
| ./main.py | # The format should now be class_name/filename.wav | `# The format should now be class_name/filename.wav` |
| ./main.py | target_filename = f"{class_name}_{uuid.uuid4().hex[:8]}.wav" | `target_filename = f"{class_name}_{uuid.uuid4().hex[:8]}.wav"` |
| ./main.py | wav_filename = f"{sound_class}_raw_{timestamp}.wav" | `wav_filename = f"{sound_class}_raw_{timestamp}.wav"` |
| ./main.py | wav_path = os.path.join(raw_sounds_dir, wav_filename) | `wav_path = os.path.join(raw_sounds_dir, wav_filename)` |
| ./main.py | audio.export(wav_path, format="wav") | `audio.export(wav_path, format="wav")` |
| ./main.py | app.logger.info(f"Converted audio to WAV: {wav_path}") | `app.logger.info(f"Converted audio to WAV: {wav_path}")` |
| ./main.py | file_path=wav_path, | `file_path=wav_path,` |
| ./main.py | new_filename = f"{class_name}_{username}_{timestamp_now}_{uuid.uuid4().hex[:6]}.wav" | `new_filename = f"{class_name}_{username}_{timestamp_now}_{uuid.uuid4().hex[:6]}.wav"` |
| ./src/core/audio/processor.py | from scipy.io import wavfile | `from scipy.io import wavfile` |
| ./src/core/audio/processor.py | 'total_files': len(wav_files), | `'total_files': len(wav_files),` |
| ./src/core/audio/processor.py | for wav_file in wav_files: | `for wav_file in wav_files:` |
| ./src/core/audio/processor.py | file_path = os.path.join(class_path, wav_file) | `file_path = os.path.join(class_path, wav_file)` |
| ./src/ml_routes_fixed.py | import wave | `import wave` |
| ./src/ml_routes_fixed.py | with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file: | `with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:` |
| ./src/ml_routes_fixed.py | with wave.open(temp_path, 'wb') as wf: | `with wave.open(temp_path, 'wb') as wf:` |
| ./src/test_mic.py | from scipy.io import wavfile | `from scipy.io import wavfile` |
| ./src/test_mic.py | wavfile.write("test_recording.wav", fs, recording_int) | `wavfile.write("test_recording.wav", fs, recording_int)` |
| ./src/ml/cnn_classifier.py | # Gather .wav files | `# Gather .wav files` |
| ./src/ml/sound_detector_ensemble.py | import wave | `import wave` |
| ./src/ml/sound_detector_ensemble.py | with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file: | `with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:` |
| ./src/ml/sound_detector_ensemble.py | with wave.open(temp_path, 'wb') as wf: | `with wave.open(temp_path, 'wb') as wf:` |
| ./src/ml/feature_extractor.py | audio_path (str): Path to the .wav (or other) audio file. | `audio_path (str): Path to the .wav (or other) audio file.` |
| ./src/ml/feature_extractor.py | def save_wav(self, audio_data, sr, filename): | `def save_wav(self, audio_data, sr, filename):` |
| ./src/ml/feature_extractor.py | Save audio_data (numpy array) to a .wav file at the specified sample rate. | `Save audio_data (numpy array) to a .wav file at the specified sample rate.` |
| ./src/ml/feature_extractor.py | logging.error(f"Error saving wav file {filename}: {e}") | `logging.error(f"Error saving wav file {filename}: {e}")` |
| ./src/ml/inference.py | from scipy.io import wavfile | `from scipy.io import wavfile` |
| ./src/ml/inference.py | import wave | `import wave` |
| ./src/ml/inference.py | with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file: | `with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:` |
| ./src/ml/inference.py | with wave.open(temp_path, 'wb') as wf: | `with wave.open(temp_path, 'wb') as wf:` |
| ./src/ml/inference.py | wavfile.write("test_recording.wav", TEST_FS, test_recording_int) | `wavfile.write("test_recording.wav", TEST_FS, test_recording_int)` |
| ./src/ml/trainer.py | Gather wavefiles from goodsounds_dir and extract "classical" features | `Gather wavefiles from goodsounds_dir and extract "classical" features` |
| ./src/ml/trainer.py | logging.info(f"# NEW LOG: Sound class='{sound}' has {len(files)} .wav files. path_sound={path_sound}") | `logging.info(f"# NEW LOG: Sound class='{sound}' has {len(files)} .wav files. path_sound={path_sound}")` |
| ./src/ml/trainer.py | audio = AudioSegment.from_file(filepath, format="wav") | `audio = AudioSegment.from_file(filepath, format="wav")` |
| ./src/ml/trainer.py | base_name = os.path.basename(filepath).replace('.wav', '_preprocessed.wav') | `base_name = os.path.basename(filepath).replace('.wav', '_preprocessed.wav')` |
| ./src/ml/trainer.py | audio.export(temp_path, format="wav") | `audio.export(temp_path, format="wav")` |
| ./src/ml/sound_detector_rf.py | import wave | `import wave` |
| ./src/ml/sound_detector_rf.py | with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file: | `with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:` |
| ./src/ml/sound_detector_rf.py | with wave.open(temp_path, 'wb') as wf: | `with wave.open(temp_path, 'wb') as wf:` |
| ./src/api/dashboard_api.py | 'icon': 'soundwave', | `'icon': 'soundwave',` |
| ./src/audio_chunker.py | from scipy.io import wavfile | `from scipy.io import wavfile` |
| ./src/audio_chunker.py | Splits a single wav file into smaller chunks based on silence detection, | `Splits a single wav file into smaller chunks based on silence detection,` |
| ./src/audio_chunker.py | rate, data = wavfile.read(filename) | `rate, data = wavfile.read(filename)` |
| ./src/audio_chunker.py | chunk_filename = filename.replace('.wav', f'_chunk_{i}.wav') | `chunk_filename = filename.replace('.wav', f'_chunk_{i}.wav')` |
| ./src/audio_chunker.py | wavfile.write(filename, rate, data_out.astype(np.int16)) | `wavfile.write(filename, rate, data_out.astype(np.int16))` |
| ./src/templates/sounds_record.html | .waveform-container { | `.waveform-container {` |
| ./src/templates/sounds_record.html | <i class="bi bi-soundwave me-2"></i>Step 2: Record Sound | `<i class="bi bi-soundwave me-2"></i>Step 2: Record Sound` |
| ./src/templates/sounds_record.html | <i class="bi bi-soundwave me-1"></i>Check Ambient Noise | `<i class="bi bi-soundwave me-1"></i>Check Ambient Noise` |
| ./src/templates/sounds_record.html | <div id="waveform" class="my-4"></div> | `<div id="waveform" class="my-4"></div>` |
| ./src/templates/sounds_record.html | formData.append('audio', blob, 'recording.wav'); | `formData.append('audio', blob, 'recording.wav');` |
| ./src/templates/base.html | <i class="bi bi-soundwave me-2"></i>SoundsEasy | `<i class="bi bi-soundwave me-2"></i>SoundsEasy` |
| ./src/templates/base.html | <a class="nav-link" href="/predict"><i class="bi bi-soundwave me-1"></i>Predict</a> | `<a class="nav-link" href="/predict"><i class="bi bi-soundwave me-1"></i>Predict</a>` |
| ./src/templates/predict.html | .waveform-container { | `.waveform-container {` |
| ./src/templates/predict.html | <i class="bi bi-soundwave me-2"></i>Real-Time Predictor | `<i class="bi bi-soundwave me-2"></i>Real-Time Predictor` |
| ./src/templates/predict.html | <div class="waveform-container" id="waveform"> | `<div class="waveform-container" id="waveform">` |
| ./src/templates/predict.html | <h5><i class="bi bi-soundwave me-2"></i>Latest Prediction</h5> | `<h5><i class="bi bi-soundwave me-2"></i>Latest Prediction</h5>` |
| ./src/templates/predict.html | <script src="https://unpkg.com/wavesurfer.js@6.6.3/dist/wavesurfer.js"></script> | `<script src="https://unpkg.com/wavesurfer.js@6.6.3/dist/wavesurfer.js"></script>` |
| ./src/templates/predict.html | let wavesurfer; | `let wavesurfer;` |
| ./src/templates/predict.html | // Initialize wavesurfer | `// Initialize wavesurfer` |
| ./src/templates/predict.html | wavesurfer = WaveSurfer.create({ | `wavesurfer = WaveSurfer.create({` |
| ./src/templates/predict.html | container: '#waveform', | `container: '#waveform',` |
| ./src/templates/predict.html | waveColor: 'rgba(67, 97, 238, 0.3)', | `waveColor: 'rgba(67, 97, 238, 0.3)',` |
| ./src/templates/register.html | <i class="bi bi-soundwave"></i> | `<i class="bi bi-soundwave"></i>` |
| ./src/templates/login.html | <i class="bi bi-soundwave"></i> | `<i class="bi bi-soundwave"></i>` |
| ./src/templates/sound_class_view.html | .waveform-container { | `.waveform-container {` |
| ./src/templates/sound_class_view.html | <div class="waveform-container" id="waveform"> | `<div class="waveform-container" id="waveform">` |
| ./src/templates/sound_class_view.html | <script src="https://unpkg.com/wavesurfer.js@6.6.3/dist/wavesurfer.js"></script> | `<script src="https://unpkg.com/wavesurfer.js@6.6.3/dist/wavesurfer.js"></script>` |
| ./src/templates/sound_class_view.html | let wavesurfer; | `let wavesurfer;` |
| ./src/templates/sound_class_view.html | // Initialize wavesurfer and other functionality | `// Initialize wavesurfer and other functionality` |
| ./src/templates/sound_class_view.html | wavesurfer = WaveSurfer.create({ | `wavesurfer = WaveSurfer.create({` |
| ./src/templates/sound_class_view.html | container: '#waveform', | `container: '#waveform',` |
| ./src/templates/sound_class_view.html | waveColor: 'rgba(67, 97, 238, 0.3)', | `waveColor: 'rgba(67, 97, 238, 0.3)',` |
| ./src/templates/sound_class_view.html | // Create audio blob and set up wavesurfer | `// Create audio blob and set up wavesurfer` |
| ./src/templates/sound_class_view.html | audioBlob = new Blob(recordedChunks, { type: 'audio/wav' }); | `audioBlob = new Blob(recordedChunks, { type: 'audio/wav' });` |
| ./src/templates/sound_class_view.html | wavesurfer.load(audioUrl); | `wavesurfer.load(audioUrl);` |
| ./src/templates/sound_class_view.html | wavesurfer.playPause(); | `wavesurfer.playPause();` |
| ./src/templates/sound_class_view.html | `recording_${new Date().toISOString().replace(/[:.-]/g, '_')}.wav`; | ``recording_${new Date().toISOString().replace(/[:.-]/g, '_')}.wav`;` |
| ./src/templates/sound_class_view.html | // Reset wavesurfer | `// Reset wavesurfer` |
| ./src/templates/sound_class_view.html | wavesurfer.empty(); | `wavesurfer.empty();` |
| ./src/templates/dashboard.html | <i class="bi bi-soundwave"></i> | `<i class="bi bi-soundwave"></i>` |
| ./src/templates/sounds_management.html | <div class="waveform-container" id="waveform"> | `<div class="waveform-container" id="waveform">` |
| ./src/templates/sounds_management.html | <script src="https://unpkg.com/wavesurfer.js@6.6.3/dist/wavesurfer.js"></script> | `<script src="https://unpkg.com/wavesurfer.js@6.6.3/dist/wavesurfer.js"></script>` |
| ./src/templates/sounds_management.html | let wavesurfer; | `let wavesurfer;` |
| ./src/templates/sounds_management.html | // Initialize wavesurfer | `// Initialize wavesurfer` |
| ./src/templates/sounds_management.html | wavesurfer = WaveSurfer.create({ | `wavesurfer = WaveSurfer.create({` |
| ./src/templates/sounds_management.html | container: '#waveform', | `container: '#waveform',` |
| ./src/templates/sounds_management.html | waveColor: 'rgba(67, 97, 238, 0.3)', | `waveColor: 'rgba(67, 97, 238, 0.3)',` |
| ./src/templates/sounds_management.html | // Create audio blob and set up wavesurfer | `// Create audio blob and set up wavesurfer` |
| ./src/templates/sounds_management.html | audioBlob = new Blob(recordedChunks, { type: 'audio/wav' }); | `audioBlob = new Blob(recordedChunks, { type: 'audio/wav' });` |
| ./src/templates/sounds_management.html | wavesurfer.load(audioUrl); | `wavesurfer.load(audioUrl);` |
| ./src/templates/sounds_management.html | wavesurfer.playPause(); | `wavesurfer.playPause();` |
| ./src/templates/sounds_management.html | `recording_${new Date().toISOString().replace(/[:.-]/g, '_')}.wav`; | ``recording_${new Date().toISOString().replace(/[:.-]/g, '_')}.wav`;` |
| ./src/templates/sounds_management.html | // Reset wavesurfer | `// Reset wavesurfer` |
| ./src/templates/sounds_management.html | wavesurfer.empty(); | `wavesurfer.empty();` |
| ./src/templates/dictionary_view.html | .waveform-container { | `.waveform-container {` |
| ./src/templates/dictionary_view.html | <div class="waveform-container" id="waveform"> | `<div class="waveform-container" id="waveform">` |
| ./src/templates/dictionary_view.html | <label for="audioFiles" class="form-label">Audio Files (.wav only)</label> | `<label for="audioFiles" class="form-label">Audio Files (.wav only)</label>` |
| ./src/templates/dictionary_view.html | <input type="file" class="form-control" id="audioFiles" accept=".wav" multiple required> | `<input type="file" class="form-control" id="audioFiles" accept=".wav" multiple required>` |
| ./src/templates/dictionary_view.html | <script src="https://unpkg.com/wavesurfer.js@6.6.3/dist/wavesurfer.js"></script> | `<script src="https://unpkg.com/wavesurfer.js@6.6.3/dist/wavesurfer.js"></script>` |
| ./src/templates/dictionary_view.html | let wavesurfer; | `let wavesurfer;` |
| ./src/templates/dictionary_view.html | // Initialize wavesurfer | `// Initialize wavesurfer` |
| ./src/templates/dictionary_view.html | wavesurfer = WaveSurfer.create({ | `wavesurfer = WaveSurfer.create({` |
| ./src/templates/dictionary_view.html | container: '#waveform', | `container: '#waveform',` |
| ./src/templates/dictionary_view.html | waveColor: 'rgba(67, 97, 238, 0.3)', | `waveColor: 'rgba(67, 97, 238, 0.3)',` |
| ./src/templates/dictionary_view.html | // Create audio blob and set up wavesurfer | `// Create audio blob and set up wavesurfer` |
| ./src/templates/dictionary_view.html | audioBlob = new Blob(recordedChunks, { type: 'audio/wav' }); | `audioBlob = new Blob(recordedChunks, { type: 'audio/wav' });` |
| ./src/templates/dictionary_view.html | wavesurfer.load(audioUrl); | `wavesurfer.load(audioUrl);` |
| ./src/templates/dictionary_view.html | wavesurfer.playPause(); | `wavesurfer.playPause();` |
| ./src/templates/dictionary_view.html | `recording_${new Date().toISOString().replace(/[:.-]/g, '_')}.wav`; | ``recording_${new Date().toISOString().replace(/[:.-]/g, '_')}.wav`;` |
| ./src/templates/dictionary_view.html | // Reset wavesurfer | `// Reset wavesurfer` |
| ./src/templates/dictionary_view.html | wavesurfer.empty(); | `wavesurfer.empty();` |
| ./src/templates/upload_sounds.html | 2. Choose one or more .wav files to upload<br> | `2. Choose one or more .wav files to upload<br>` |
| ./src/templates/upload_sounds.html | accept=".wav" | `accept=".wav"` |
| ./src/templates/upload_sounds.html | You can select multiple .wav files at once | `You can select multiple .wav files at once` |
| ./src/templates/verify.html | <i class="bi bi-soundwave me-2"></i> | `<i class="bi bi-soundwave me-2"></i>` |
| ./src/routes/ml_routes.py | wavfile.write(temp_path, 16000, np.int16(trimmed_data * 32767)) | `wavfile.write(temp_path, 16000, np.int16(trimmed_data * 32767))` |
| ./src/routes/ml_routes.py | wav_data, _ = librosa.load(temp_path, sr=16000) | `wav_data, _ = librosa.load(temp_path, sr=16000)` |
| ./src/routes/ml_routes.py | cnn_features = sp.process_audio(wav_data) | `cnn_features = sp.process_audio(wav_data)` |
| ./src/routes/ml_routes.py | # Convert webm->wav with pydub | `# Convert webm->wav with pydub` |
| ./src/routes/ml_routes.py | wav_io = io.BytesIO() | `wav_io = io.BytesIO()` |
| ./src/routes/ml_routes.py | audio.export(wav_io, format="wav") | `audio.export(wav_io, format="wav")` |
| ./src/routes/ml_routes.py | wav_io.seek(0) | `wav_io.seek(0)` |
| ./src/routes/ml_routes.py | temp_filename = f"{sound}_{session['username']}_{timestamp}.wav" | `temp_filename = f"{sound}_{session['username']}_{timestamp}.wav"` |
| ./src/routes/ml_routes.py | f.write(wav_io.read()) | `f.write(wav_io.read())` |
| ./src/routes/ml_routes.py | new_filename = f"{sound}_{username}_{existing_count + 1}.wav" | `new_filename = f"{sound}_{username}_{existing_count + 1}.wav"` |
| ./src/routes/ml_routes.py | f"{sound}_{session['username']}_{timestamp}_temp.wav") | `f"{sound}_{session['username']}_{timestamp}_temp.wav")` |
| ./src/services/dictionary_service.py | # Ensure sample has .wav extension | `# Ensure sample has .wav extension` |
| ./src/services/dictionary_service.py | sample_name += '.wav' | `sample_name += '.wav'` |
| ./src/services/dictionary_service.py | # Get all .wav files | `# Get all .wav files` |
| ./src/services/dictionary_service.py | # Get all .wav files | `# Get all .wav files` |
| ./src/services/training_service.py | if not wav_files: | `if not wav_files:` |
| ./src/services/training_service.py | error_msg = f"No .wav files found in {class_path}" | `error_msg = f"No .wav files found in {class_path}"` |
| ./src/services/training_service.py | stats['original_counts'][class_dir] = len(wav_files) | `stats['original_counts'][class_dir] = len(wav_files)` |
| ./src/services/training_service.py | logging.info(f"Processing class {class_dir}: Found {len(wav_files)} audio files") | `logging.info(f"Processing class {class_dir}: Found {len(wav_files)} audio files")` |
| ./src/services/training_service.py | self.error_logs.append({'level': 'INFO', 'message': f"Processing class {class_dir}: Found {len(wav_files)} audio files"}) | `self.error_logs.append({'level': 'INFO', 'message': f"Processing class {class_dir}: Found {len(wav_files)} audio files"})` |
| ./src/services/training_service.py | for wav_file in wav_files: | `for wav_file in wav_files:` |
| ./src/services/training_service.py | file_path = os.path.join(class_path, wav_file) | `file_path = os.path.join(class_path, wav_file)` |
| ./src/services/training_service.py | if not wav_files: | `if not wav_files:` |
| ./src/services/training_service.py | logging.warning(f"No .wav files found in {class_path}") | `logging.warning(f"No .wav files found in {class_path}")` |
| ./src/services/training_service.py | stats['original_counts'][class_dir] = len(wav_files) | `stats['original_counts'][class_dir] = len(wav_files)` |
| ./src/services/training_service.py | for wav_file in wav_files: | `for wav_file in wav_files:` |
| ./src/services/training_service.py | file_path = os.path.join(class_path, wav_file) | `file_path = os.path.join(class_path, wav_file)` |
| ./src/services/training_service.py | if len(wav_files) < min_samples_per_class: | `if len(wav_files) < min_samples_per_class:` |
| ./src/services/training_service.py | insufficient_classes.append((class_dir, len(wav_files))) | `insufficient_classes.append((class_dir, len(wav_files)))` |

