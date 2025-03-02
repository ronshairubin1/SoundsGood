# SoundClassifier_v08/src/ml/cnn_classifier.py

import os
import numpy as np
import random
import librosa
import logging
import json
from io import StringIO

import tensorflow as tf
from tensorflow.keras import layers, models

from config import Config
from .audio_processing import SoundProcessor
from .constants import (
    SAMPLE_RATE, AUDIO_DURATION, AUDIO_LENGTH, BATCH_SIZE, EPOCHS,
    AUG_DO_TIME_SHIFT, AUG_TIME_SHIFT_COUNT, AUG_SHIFT_MAX,
    AUG_DO_PITCH_SHIFT, AUG_PITCH_SHIFT_COUNT, AUG_PITCH_RANGE,
    AUG_DO_SPEED_CHANGE, AUG_SPEED_CHANGE_COUNT, AUG_SPEED_RANGE,
    AUG_DO_NOISE, AUG_NOISE_COUNT, AUG_NOISE_TYPE, AUG_NOISE_FACTOR,
    PITCH_SHIFTS_OUTER_VALUES, PITCH_SHIFTS_CENTER_START, PITCH_SHIFTS_CENTER_END, PITCH_SHIFTS_CENTER_NUM,
    NOISE_TYPES_LIST, NOISE_LEVELS_MIN, NOISE_LEVELS_MAX, NOISE_LEVELS_COUNT,
    AUG_DO_COMPRESSION, AUG_COMPRESSION_COUNT,
    AUG_DO_REVERB, AUG_REVERB_COUNT,
    AUG_DO_EQUALIZE, AUG_EQUALIZE_COUNT 
)

from .augmentation_manager import augment_audio_with_repetitions

"""
Merged CNN Classifier code with data augmentation, 
consistent references to Config and SoundProcessor.
"""

# -----------------------------
# Main function to build dataset
# -----------------------------
def build_dataset(sound_folder):
    """
    Build a dataset for CNN training by loading audio from
    `sound_folder` (the 'goodsounds' directory) and using the
    active dictionary from config. Applies data augmentation
    to each original file.
    
    Returns:
        X: np.array of shape (N, 63, 64, 1) [example shape]
        y: np.array of labels
        class_names: list of sound labels
        stats: dict with info on original/augmented counts
    """
    X = []
    y = []
    total_samples = 0

    # Initialize SoundProcessor
    sound_processor = SoundProcessor(sample_rate=SAMPLE_RATE)

    # Load the active dictionary from config
    config_file = os.path.join(Config.CONFIG_DIR, 'active_dictionary.json')
    logging.info(f"Looking for config file at: {config_file}")
    with open(config_file, 'r') as f:
        active_dict = json.load(f)
    class_names = active_dict['sounds']
    logging.info(f"Found class names: {class_names}")

    # Map each class_name to an integer
    class_indices = {name: i for i, name in enumerate(class_names)}
    logging.info(f"Class indices mapping: {class_indices}")

    # Example pitch shifting steps and noise arrays from constants.py
    pitch_shifts_center = np.linspace(
        PITCH_SHIFTS_CENTER_START,
        PITCH_SHIFTS_CENTER_END,
        PITCH_SHIFTS_CENTER_NUM
    )
    pitch_shifts_outer = np.array(PITCH_SHIFTS_OUTER_VALUES)
    pitch_shifts = np.concatenate([pitch_shifts_outer, pitch_shifts_center])

    noise_types = NOISE_TYPES_LIST
    noise_levels = np.linspace(NOISE_LEVELS_MIN, NOISE_LEVELS_MAX, NOISE_LEVELS_COUNT)

    # For each sound in the active dictionary
    for class_name in class_names:
        class_path = os.path.join(sound_folder, class_name)
        logging.info(f"Processing class directory: {class_path}")
        if not os.path.exists(class_path):
            logging.warning(f"Directory {class_path} does not exist.")
            continue

        # Gather .wav files
        files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
        logging.info(f"Found {len(files)} files for class {class_name}")

        for file_name in files:
            file_path = os.path.join(class_path, file_name)
            try:
                logging.info(f"Loading file: {file_path}")
                audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)

                # Original
                features = sound_processor.process_audio(audio)
                if features is None:
                    continue
                X.append(features)
                y.append(class_indices[class_name])
                total_samples += 1

                # Multi-run augmentation with explicit parameter names:
                augmented_clips = augment_audio_with_repetitions(
                    audio=audio,
                    sr=SAMPLE_RATE,
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
                for aug_audio in augmented_clips:
                    features_aug = sound_processor.process_audio(aug_audio)
                    if features_aug is not None:
                        X.append(features_aug)
                        y.append(class_indices[class_name])
                        total_samples += 1

            except Exception as e:
                logging.error(f"Error processing {file_path}: {e}")
                continue

    # Final arrays
    if not X:
        logging.error("No valid samples were processed.")
        return None, None, None, None

    X = np.array(X)
    y = np.array(y)
    logging.info(f"Total samples after augmentation: {total_samples}")
    logging.info(f"Final dataset shapes: X={X.shape}, y={y.shape}")

    # Minimal stats
    stats = {
        'original_counts': {},
        'augmented_counts': {}
    }
    # You can fill these if needed for advanced logging

    return X, y, class_names, stats

# -----------------------------
# Build CNN model
# -----------------------------
def build_model(input_shape, num_classes):
    """
    Build a CNN model optimized for short audio classification.
    Returns (model, model_summary_string).
    """
    inputs = layers.Input(shape=input_shape)

    # First Conv Block
    x = layers.Conv2D(16, (3, 3), padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(1e-5))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    # Second Conv Block
    x = layers.Conv2D(32, (3, 3), padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)

    # Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(1e-5))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    # Adam with a lower LR
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Capture the summary
    summary_io = StringIO()
    model.summary(print_fn=lambda s: summary_io.write(s + '\n'))
    model_summary = summary_io.getvalue()

    return model, model_summary

def gather_cnn_summary_data(X_train, y_train, X_val, y_val, class_names, dataset_stats, model_summary_str, keras_history):
    # Basic dictionary of stats
    training_stats = {}
    training_history = {}

    # Input shape
    if X_train is not None and len(X_train.shape) > 1:
        training_stats['input_shape'] = list(X_train.shape[1:])
    else:
        training_stats['input_shape'] = None

    # Total samples in training + validation
    total_train = len(X_train) if X_train is not None else 0
    total_val = len(X_val) if X_val is not None else 0
    training_stats['total_samples'] = total_train + total_val

    # Classes
    training_stats['classes'] = class_names

    # Any dataset stats you collected, e.g. MFCC range or other
    if dataset_stats:
        training_stats['mfcc_range'] = dataset_stats.get('mfcc_range', None)
    else:
        training_stats['mfcc_range'] = None

    # The string summary from build_model
    training_stats['model_summary'] = model_summary_str

    # If Keras history is available, compute final validation accuracy
    if hasattr(keras_history, 'history'):
        val_acc_list = keras_history.history.get('val_accuracy', [])
        if val_acc_list:
            training_stats['cnn_val_accuracy'] = val_acc_list[-1]
        else:
            training_stats['cnn_val_accuracy'] = None

        # training_history is used by your template to show or plot epoch-wise data
        training_history['epochs'] = len(keras_history.history.get('loss', []))
        training_history['loss'] = keras_history.history.get('loss', [])
        training_history['val_loss'] = keras_history.history.get('val_loss', [])
        training_history['accuracy'] = keras_history.history.get('accuracy', [])
        training_history['val_accuracy'] = keras_history.history.get('val_accuracy', [])
    else:
        training_stats['cnn_val_accuracy'] = None
        training_history = None

    return training_stats, training_history

def gather_training_stats_for_cnn():
    """
    Placeholder function you can call from ml_routes.py if needed.
    It might load the stats from a global variable, a database, or from disk.
    For now, it just returns an empty dict.
    """
    # For example, you might do:
    # return global_cnn_training_stats, global_cnn_history
    return {}, {}

def get_cnn_history():
    """
    Placeholder function if you want to separately retrieve just the training_history.
    For now, returns an empty dict.
    """
    # For example, you might do:
    # return global_cnn_history
    return {}

# -----------------------------
# Optional: training test
# -----------------------------
if __name__ == "__main__":
    """
    If you want to do a quick test run from the command line:
    python cnn_classifier.py
    (You might have to adjust data_path or feed real audio.)
    """
    data_path = "data"  # adjust to your actual data folder

    # Build dataset
    X, y, class_names, stats = build_dataset(data_path)
    if X is None or y is None:
        print("No data found. Exiting.")
        exit(0)

    # Shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Split
    val_split = 0.2
    split_idx = int(len(X) * (1 - val_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Build the model
    input_shape = X_train.shape[1:]  # e.g. (63, 64, 1)
    model, model_summary = build_model(input_shape, num_classes=len(class_names))
    print(model_summary)

    # Example training
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        min_delta=0.001,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[early_stopping]
    )

    # Save the model
    model.save("audio_classifier.h5")

    # Evaluate
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation accuracy: {val_acc*100:.2f}%")
