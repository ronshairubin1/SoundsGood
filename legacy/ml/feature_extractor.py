# SoundClassifier_v08/src/ml/feature_extractor.py

import librosa
import numpy as np
import logging

class AudioFeatureExtractor:
    """
    A consolidated feature extractor that merges old_code/feature_extractor.py 
    style MFCC + spectral features with optional formants/pitch. 
    This can be used by the RandomForest approach or for 
    a 'classical' feature-based method.
    """
    def __init__(self, sr=22050, duration=None):
        """
        Args:
            sr (int): Sample rate for audio processing
            duration (float): Duration to load from audio file (None for full file)
        """
        self.sr = sr
        self.duration = duration
        self.n_mfcc = 13
        self.hop_length = 512

    def extract_features(self, audio_path):
        """
        Extract features from an audio file on disk.
        
        Args:
            audio_path (str): Path to the .wav (or other) audio file.
        Returns:
            dict or None: A dictionary of extracted features, or None if there's an error.
        """
        try:
            # Load audio from file
            y, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration)

            # Time-stretch short audio to ensure at least 9 frames
            required_len = self.hop_length * 9
            if len(y) < required_len:
                stretch_factor = 1.05 * required_len / len(y)
                logging.warning(
                    f"Audio too short ({len(y)} < {required_len}). "
                    f"Time-stretching by factor={stretch_factor:.2f}"
                )
                y = librosa.effects.time_stretch(y, rate=1 / stretch_factor)
                logging.warning(f"Audio now {len(y)} samples!")

            features = {}

            # --------------------
            # MFCC + deltas 
            # --------------------
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length)
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

            for i in range(self.n_mfcc):
                features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
                features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
                features[f'mfcc_delta_{i}_mean'] = float(np.mean(mfcc_delta[i]))
                features[f'mfcc_delta_{i}_std'] = float(np.std(mfcc_delta[i]))
                features[f'mfcc_delta2_{i}_mean'] = float(np.mean(mfcc_delta2[i]))
                features[f'mfcc_delta2_{i}_std'] = float(np.std(mfcc_delta2[i]))

            # --------------------
            # Formant approximation
            # --------------------
            formants = librosa.effects.preemphasis(y)
            features['formant_mean'] = float(np.mean(formants))
            features['formant_std'] = float(np.std(formants))

            # --------------------
            # Pitch
            # --------------------
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_vals = pitches[magnitudes > np.median(magnitudes)]
            if len(pitch_vals) > 0:
                features['pitch_mean'] = float(np.mean(pitch_vals))
                features['pitch_std'] = float(np.std(pitch_vals))
            else:
                features['pitch_mean'] = 0.0
                features['pitch_std'] = 0.0

            # --------------------
            # Spectral Centroid
            # --------------------
            cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid_mean'] = float(np.mean(cent))
            features['spectral_centroid_std'] = float(np.std(cent))

            # --------------------
            # Zero Crossing Rate
            # --------------------
            zcr = librosa.feature.zero_crossing_rate(y)
            features['zcr_mean'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))

            # --------------------
            # RMS
            # --------------------
            rms = librosa.feature.rms(y=y)
            features['rms_mean'] = float(np.mean(rms))
            features['rms_std'] = float(np.std(rms))

            # --------------------
            # Spectral Rolloff
            # --------------------
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features['rolloff_mean'] = float(np.mean(rolloff))
            features['rolloff_std'] = float(np.std(rolloff))

            return features
        except Exception as e:
            logging.error(f"Error extracting features from {audio_path}: {str(e)}")
            return None

    def extract_features_from_array(self, y, sr=None):
        """
        Extract features from a raw numpy array (y) already in memory.
        
        Args:
            y (np.ndarray): Raw audio array.
            sr (int, optional): If provided, overrides self.sr for this extraction.
        Returns:
            dict or None: A dictionary of extracted features, or None if there's an error.
        """
        try:
            # Use the passed sr if present, otherwise self.sr
            if sr is None:
                sr = self.sr

            # Time-stretch short audio to ensure at least 9 frames
            required_len = self.hop_length * 9
            if len(y) < required_len:
                stretch_factor = 1.05 * required_len / len(y)
                logging.warning(
                    f"Audio too short ({len(y)} < {required_len}). "
                    f"Time-stretching by factor={stretch_factor:.2f}"
                )
                y = librosa.effects.time_stretch(y, rate=1 / stretch_factor)

            features = {}

            # --------------------
            # MFCC + deltas 
            # --------------------
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length)
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

            for i in range(self.n_mfcc):
                features[f'mfcc_{i}_mean'] = float(np.mean(mfccs[i]))
                features[f'mfcc_{i}_std'] = float(np.std(mfccs[i]))
                features[f'mfcc_delta_{i}_mean'] = float(np.mean(mfcc_delta[i]))
                features[f'mfcc_delta_{i}_std'] = float(np.std(mfcc_delta[i]))
                features[f'mfcc_delta2_{i}_mean'] = float(np.mean(mfcc_delta2[i]))
                features[f'mfcc_delta2_{i}_std'] = float(np.std(mfcc_delta2[i]))

            # --------------------
            # Formant approximation
            # --------------------
            formants = librosa.effects.preemphasis(y)
            features['formant_mean'] = float(np.mean(formants))
            features['formant_std'] = float(np.std(formants))

            # --------------------
            # Pitch
            # --------------------
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
            pitch_vals = pitches[magnitudes > np.median(magnitudes)]
            if len(pitch_vals) > 0:
                features['pitch_mean'] = float(np.mean(pitch_vals))
                features['pitch_std'] = float(np.std(pitch_vals))
            else:
                features['pitch_mean'] = 0.0
                features['pitch_std'] = 0.0

            # --------------------
            # Spectral Centroid
            # --------------------
            cent = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid_mean'] = float(np.mean(cent))
            features['spectral_centroid_std'] = float(np.std(cent))

            # --------------------
            # Zero Crossing Rate
            # --------------------
            zcr = librosa.feature.zero_crossing_rate(y)
            features['zcr_mean'] = float(np.mean(zcr))
            features['zcr_std'] = float(np.std(zcr))

            # --------------------
            # RMS
            # --------------------
            rms = librosa.feature.rms(y=y)
            features['rms_mean'] = float(np.mean(rms))
            features['rms_std'] = float(np.std(rms))

            # --------------------
            # Spectral Rolloff
            # --------------------
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features['rolloff_mean'] = float(np.mean(rolloff))
            features['rolloff_std'] = float(np.std(rolloff))

            return features
        except Exception as e:
            logging.error(f"Error extracting features from raw audio array: {e}")
            return None

    def get_feature_names(self):
        """
        Return a list of feature names in the order they appear
        in the extracted feature dictionaries.
        """
        feature_names = []
        for i in range(self.n_mfcc):
            feature_names.extend([
                f'mfcc_{i}_mean', f'mfcc_{i}_std',
                f'mfcc_delta_{i}_mean', f'mfcc_delta_{i}_std',
                f'mfcc_delta2_{i}_mean', f'mfcc_delta2_{i}_std'
            ])
        feature_names.extend([
            'formant_mean', 'formant_std',
            'pitch_mean', 'pitch_std',
            'spectral_centroid_mean', 'spectral_centroid_std',
            'zcr_mean', 'zcr_std',
            'rms_mean', 'rms_std',
            'rolloff_mean', 'rolloff_std'
        ])
        return feature_names

    def save_wav(self, audio_data, sr, filename):
        """
        Save audio_data (numpy array) to a .wav file at the specified sample rate.
        """
        import soundfile as sf
        try:
            sf.write(filename, audio_data, sr)
        except Exception as e:
            logging.error(f"Error saving wav file {filename}: {e}")
