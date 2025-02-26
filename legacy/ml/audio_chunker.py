# File: SoundClassifier_v08/src/audio_chunker.py

import os
import numpy as np
from scipy.io import wavfile

class SoundProcessor:
    def __init__(self):
        self.min_chunk_duration = 0.2
        self.silence_threshold = 0.1
        self.min_silence_duration = 0.1
        self.max_silence_duration = 2.0

    def chop_recording(self, filename):
        """
        Splits a single wav file into smaller chunks based on silence detection,
        saving them to disk, then returning the list of chunk filenames.
        """
        print(f"Processing file: {filename}")
        rate, data = wavfile.read(filename)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        data = data / np.max(np.abs(data))

        is_silence = np.abs(data) < self.silence_threshold
        print(f"Found {np.sum(~is_silence)} non-silent samples out of {len(data)}")
        print(f"Silence threshold: {self.silence_threshold}")

        silence_starts = []
        silence_ends = []
        current_silence_start = None

        for i, val in enumerate(is_silence):
            if val and current_silence_start is None:
                current_silence_start = i
            elif not val and current_silence_start is not None:
                silence_duration = (i - current_silence_start) / rate
                if self.min_silence_duration <= silence_duration <= self.max_silence_duration:
                    silence_starts.append(current_silence_start)
                    silence_ends.append(i)
                    print(f"Found silence: {silence_duration:.2f}s")
                current_silence_start = None

        print(f"Found {len(silence_starts)} valid silences")

        chunk_starts = []
        chunk_ends = []
        if not silence_starts and (len(data)/rate > self.min_chunk_duration):
            chunk_starts = [0]
            chunk_ends = [len(data)]
        else:
            if silence_starts:
                chunk_starts.append(0)
            for s, e in zip(silence_starts, silence_ends):
                chunk_ends.append(s)
                chunk_starts.append(e)
            if silence_ends:
                chunk_ends.append(len(data))
        
        chunk_files = []
        for i, (start, end) in enumerate(zip(chunk_starts, chunk_ends)):
            duration = (end - start) / rate
            print(f"Chunk {i}: duration = {duration:.2f}s")
            if duration > self.min_chunk_duration:
                chunk_filename = filename.replace('.wav', f'_chunk_{i}.wav')
                self._save_chunk(data[start:end], rate, chunk_filename)
                chunk_files.append(os.path.basename(chunk_filename))
            else:
                print(f"Rejecting chunk {i}: too short ({duration:.2f}s)")

        return chunk_files

    def _save_chunk(self, data, rate, filename):
        data_out = data * 32767
        wavfile.write(filename, rate, data_out.astype(np.int16))
