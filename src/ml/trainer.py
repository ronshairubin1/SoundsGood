# SoundClassifier_v08/src/ml/trainer.py

import os
import logging
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from .rf_classifier import RandomForestClassifier
from .cnn_classifier import build_model, build_dataset
from .feature_extractor import AudioFeatureExtractor
from .audio_processing import SoundProcessor
from config import Config
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from .data_augmentation import (
    time_shift,
    change_pitch,
    change_speed,
    add_colored_noise
)
from .constants import (
    AUG_DO_TIME_SHIFT, AUG_TIME_SHIFT_COUNT, AUG_SHIFT_MAX,
    AUG_DO_PITCH_SHIFT, AUG_PITCH_SHIFT_COUNT, AUG_PITCH_RANGE,
    AUG_DO_SPEED_CHANGE, AUG_SPEED_CHANGE_COUNT, AUG_SPEED_RANGE,
    AUG_DO_NOISE, AUG_NOISE_COUNT, AUG_NOISE_TYPE, AUG_NOISE_FACTOR,
    PITCH_SHIFTS_OUTER_VALUES, PITCH_SHIFTS_CENTER_START, 
    PITCH_SHIFTS_CENTER_END, PITCH_SHIFTS_CENTER_NUM,
    NOISE_TYPES_LIST, NOISE_LEVELS_MIN, NOISE_LEVELS_MAX, NOISE_LEVELS_COUNT,
    AUG_DO_COMPRESSION, AUG_COMPRESSION_COUNT,
    AUG_DO_REVERB, AUG_REVERB_COUNT,
    AUG_DO_EQUALIZE, AUG_EQUALIZE_COUNT
)
from .augmentation_manager import augment_audio_with_repetitions

#from .model import SoundClassifier
#import librosa
#import soundfile as sf
#from datetime import datetime
#from .sound_preprocessor import SoundPreprocessor
#from pydub import AudioSegment
#from sklearn.preprocessing import StandardScaler
#import pandas as pd
#import joblib
#import tempfile
#from tqdm import tqdm

class Trainer:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.rf_classifier = RandomForestClassifier(model_dir=model_dir)
        self.cnn_model = None
        self.class_names = None
        self.last_rf_acc = 0.0
        self._rf_class_names = []
        self._X_train = None
        self._X_val = None
        self._stats = {
            'original_counts': {},
            'augmented_counts': {}
        }

    def collect_data_for_rf(self, goodsounds_dir):
        """
        Gather wavefiles from goodsounds_dir and extract "classical" features 
        (MFCC, pitch, etc.) for RandomForest training, with multi-run data augmentation.
        """
        logging.info(f"# NEW LOG: Starting collect_data_for_rf with goodsounds_dir={goodsounds_dir}")

        extractor = AudioFeatureExtractor(sr=16000)
        X = []
        y = []
        dictionary = Config.get_dictionary()

        # Create a SoundProcessor if you need it for boundary detection
        sound_proc = SoundProcessor(sample_rate=16000)

        # Counters for logging how many samples come from original vs. augmentation
        file_count = 0
        aug_count = 0

        for sound in dictionary['sounds']:
            path_sound = os.path.join(goodsounds_dir, sound)
            if not os.path.exists(path_sound):
                continue
            files = [f for f in os.listdir(path_sound) if f.endswith('.wav')]

            logging.info(f"# NEW LOG: Sound class='{sound}' has {len(files)} .wav files. path_sound={path_sound}")

            for filename in files:
                file_path = os.path.join(path_sound, filename)
                logging.info(f"# NEW LOG: Loading file {file_path}")
                try:
                    audio_samples, sr_ = librosa.load(
                        file_path, sr=extractor.sr, duration=extractor.duration
                    )

                    # Process audio (e.g., detect sound boundaries)
                    start_idx, end_idx, has_sound = sound_proc.detect_sound_boundaries(audio_samples)
                    logging.info(f"# NEW LOG: Boundaries -> start_idx={start_idx}, end_idx={end_idx}, has_sound={has_sound}")
                    audio_samples = audio_samples[start_idx:end_idx]

                    # Original features
                    feats_orig = extractor.extract_features_from_array(audio_samples, sr=sr_)
                    if feats_orig is not None:
                        row_orig = self._assemble_feature_row(feats_orig, extractor)
                        X.append(row_orig)
                        y.append(sound)
                        file_count += 1
                        logging.info(f"# NEW LOG: Added ORIG sample for {filename} (class: {sound})")
                    else:
                        logging.warning(f"# NEW LOG: Original sample from {filename} was skipped (no features).")

                    # Multi-run augmentation with explicit defaults:
                    logging.info(f"# NEW LOG: Attempting augmentation for {filename}")
                    augmented_versions = augment_audio_with_repetitions(
                        audio=audio_samples,
                        sr=sr_,
                        do_time_shift=AUG_DO_TIME_SHIFT,
                        time_shift_count=AUG_TIME_SHIFT_COUNT,
                        shift_max=AUG_SHIFT_MAX,

                        do_pitch_shift=AUG_DO_PITCH_SHIFT,
                        pitch_shift_count=AUG_PITCH_SHIFT_COUNT,
                        pitch_range=AUG_PITCH_RANGE,

                        do_speed_change=AUG_DO_SPEED_CHANGE,
                        speed_change_count=AUG_SPEED_CHANGE_COUNT,
                        speed_range=AUG_SPEED_RANGE,

                        do_noise=AUG_DO_NOISE,
                        noise_count=AUG_NOISE_COUNT,
                        noise_type=AUG_NOISE_TYPE,
                        noise_factor=AUG_NOISE_FACTOR,

                        do_compression=AUG_DO_COMPRESSION,
                        compression_count=AUG_COMPRESSION_COUNT,

                        do_reverb=AUG_DO_REVERB,
                        reverb_count=AUG_REVERB_COUNT,

                        do_equalize=AUG_DO_EQUALIZE,
                        equalize_count=AUG_EQUALIZE_COUNT
                    )
                    logging.info(f"# NEW LOG: Created {len(augmented_versions)} augmented clips for {filename}")

                    for aug_audio in augmented_versions:
                        aug_feats = extractor.extract_features_from_array(aug_audio, sr=sr_)
                        if aug_feats is not None:
                            row_aug = self._assemble_feature_row(aug_feats, extractor)
                            X.append(row_aug)
                            y.append(sound)
                            aug_count += 1
                            logging.info(f"# NEW LOG: Added AUG sample for {filename} (class: {sound})")
                        else:
                            logging.warning(f"# NEW LOG: Augmented clip was too short/invalid for {filename}, skipping.")
                except Exception as e:
                    logging.error(f"Error processing {file_path}: {str(e)}")
                    continue

        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Final logs for clarity
        logging.info(f"Total original samples: {file_count}")
        logging.info(f"Total augmentation samples: {aug_count}")
        logging.info(f"Final dataset shape: X={X.shape}, y={y.shape}")

        return X, y

    def _assemble_feature_row(self, feats, extractor):
        """
        Helper method that replicates your 'row' building logic, 
        matching the feature_names from extractor.
        """
        feature_names = extractor.get_feature_names()
        row = []
        for fn in feature_names:
            row.append(feats.get(fn, 0.0))  # or do your custom MFCC index logic
        return row

    def train_rf(self, goodsounds_dir):
        X, y = self.collect_data_for_rf(goodsounds_dir)
        
        # Possibly store original/augmented counts in self._stats here,
        # e.g. self._stats['original_counts'] = { ... }
        #      self._stats['augmented_counts'] = { ... }

        # Train on the entire set first if you want:
        success = self.rf_classifier.train(X, y)  # your existing line
        if success:
            self.rf_classifier.save()

        # NOW do your real train/test split (80/20 as example)
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Use X_train/y_train as your "train" set
        self.rf_classifier.train(X_train, y_train)
        preds, _ = self.rf_classifier.predict(X_test)

        acc = accuracy_score(y_test, preds)
        self.last_rf_acc = acc
        logging.info(f"RF accuracy on test: {acc:.3f}")

        # ──────────────────────────────────────────────────────────
        # ADD: store these splits in instance fields so routes can use them
        self._X_train = X_train
        self._X_val = X_test  # or rename to self._X_test
        # ──────────────────────────────────────────────────────────

        if not hasattr(self, '_rf_stats'):
            self._rf_stats = {}

        # Store the class names into _rf_stats
        self._rf_stats['class_names'] = self._rf_class_names

    def get_rf_accuracy(self):
        """Return the last RF accuracy measured."""
        return self.last_rf_acc

    def get_rf_class_names(self):
        """Return your class names as a list."""
        return self._rf_class_names

    def get_rf_train_data(self):
        """
        Returns a tuple: (X_train, X_val, stats)
        so the route can see how many train + val samples we ended up with.
        """
        return (self._X_train, self._X_val, self._stats)

    def train_cnn(self, goodsounds_dir):
        """
        Train the CNN model from your existing pipeline, ensuring we call
        the same build_dataset(...) from cnn_classifier.py, which uses SoundProcessor
        for all audio transformations.
        """
        X, y, class_names, stats = build_dataset(goodsounds_dir)
        if X is None or y is None:
            logging.error("No data found for CNN training.")
            return None, None
        
        self.class_names = class_names

        # Shuffle data
        idx = np.arange(len(X))
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # Split
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        input_shape = X_train.shape[1:]
        
        # Build and compile model
        model, summary_str = build_model(input_shape, num_classes=len(class_names))
        
        checkpoint_path = os.path.join(self.model_dir, 'cnn_checkpoint.h5')
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=3, min_lr=1e-5),
            ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=32,
            epochs=30,
            callbacks=callbacks
        )
        
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        logging.info(f"CNN val_acc: {val_acc:.3f}")

        # Save final
        model_path = os.path.join(self.model_dir, "audio_classifier.h5")
        model.save(model_path)
        self.cnn_model = model
        return model, class_names

    def train_ensemble_example(self):
        """
        Example code to show how you could load both trained models 
        and use the ensemble. You will adapt as needed in your real code.
        """
        # Load trained RF
        self.rf_classifier.load(filename='rf_sound_classifier.joblib')
        # Load trained CNN
        from tensorflow.keras.models import load_model
        model_path = os.path.join(self.model_dir, "audio_classifier.h5")
        self.cnn_model = load_model(model_path)

        # Then create an EnsembleClassifier, if desired
        from .ensemble_classifier import EnsembleClassifier
        ensemble = EnsembleClassifier(self.rf_classifier, self.cnn_model, self.class_names, rf_weight=0.5)
        ensemble.predict(X_rf, X_cnn)
        # ...

    def gather_training_stats_for_cnn(self):
        """
        Returns a dictionary that contains all CNN-specific training statistics.
        For now, this is a placeholder. You may fill it with real data from your
        CNN training flow (e.g. final accuracy, class names, overfitting checks, etc.)
        """
        # Potential approach:
        # return self._cnn_stats
        # or return a global variable if you prefer
        return getattr(self, '_cnn_stats', {})

    def gather_training_stats_for_rf(self):
        """
        Returns a dictionary of Random Forest training stats (accuracy, feature importances, etc.).
        Again, you can store these in self._rf_stats or a global variable.
        """
        return getattr(self, '_rf_stats', {})

    def gather_training_stats_for_ensemble(self):
        """
        Returns a dictionary describing the ensemble approach (combined CNN & RF stats).
        Could include 'ensemble_accuracy' and more.
        """
        return getattr(self, '_ensemble_stats', {})

    def gather_all_model_stats(self):
        """
        For model comparisons, returns a dictionary that includes
        CNN, RF, Ensemble performance side-by-side.
        Could be a simple merge of the other dictionaries or a separate structure.
        """
        summary = {}
        summary.update(self.gather_training_stats_for_cnn())    # merges CNN stats
        summary.update(self.gather_training_stats_for_rf())     # merges RF stats
        summary.update(self.gather_training_stats_for_ensemble())
        # Optionally, you can add keys like 'rf_accuracy', 'cnn_val_accuracy', 'ensemble_accuracy'
        return summary

def preprocess_audio(filepath, temp_dir):
    #Preprocess audio to match inference preprocessing
    audio = AudioSegment.from_file(filepath, format="wav")
    # Normalize audio
    audio = audio.normalize()
    # Convert to mono if stereo
    if audio.channels > 1:
        audio = audio.set_channels(1)
    # Trim silence
    audio = audio.strip_silence(silence_thresh=-40)
    # Export preprocessed audio to a temporary file
    base_name = os.path.basename(filepath).replace('.wav', '_preprocessed.wav')
    temp_path = os.path.join(temp_dir, base_name)
    audio.export(temp_path, format="wav")
    return temp_path

