import sounddevice as sd
import numpy as np
import os
from scipy.io import wavfile
from ml.constants import INT16_MAX
from config import Config

duration = 5  # seconds
fs = 44100  # Sample rate

print("Recording for 5 seconds...")
recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
sd.wait()
print("Recording complete. Playing back...")

sd.play(recording, samplerate=fs)
sd.wait()

# Save the recording to a WAV file in the test sounds directory
recording_int = np.int16(recording * INT16_MAX)
test_recording_path = os.path.join(Config.TEST_SOUNDS_DIR, "test_recording.wav")
wavfile.write(test_recording_path, fs, recording_int)
print(f"Playback complete. Recording saved to {test_recording_path}")
