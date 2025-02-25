import numpy as np
import librosa
import logging
from .data_augmentation import (
    time_shift,
    change_pitch,
    change_speed,
    add_colored_noise,
    dynamic_range_compression,
    add_reverb,
    equalize
)
from .constants import (
    AUG_DO_TIME_SHIFT, AUG_TIME_SHIFT_COUNT, AUG_SHIFT_MAX,
    AUG_DO_PITCH_SHIFT, AUG_PITCH_SHIFT_COUNT, AUG_PITCH_RANGE,
    AUG_DO_SPEED_CHANGE, AUG_SPEED_CHANGE_COUNT, AUG_SPEED_RANGE,
    AUG_DO_NOISE, AUG_NOISE_COUNT, AUG_NOISE_TYPE, AUG_NOISE_FACTOR,
    AUG_DO_COMPRESSION, AUG_COMPRESSION_COUNT,
    AUG_DO_REVERB, AUG_REVERB_COUNT,
    AUG_DO_EQUALIZE, AUG_EQUALIZE_COUNT
)

def augment_audio(
    audio, 
    sr, 
    do_time_shift=True, 
    do_pitch_shift=True, 
    do_speed_change=True,
    do_noise=True,
    noise_type='white',
    noise_factor=0.005,
    pitch_range=2.0
):
    """
    Takes an original audio array and applies one or more data augmentations,
    returning a list of new augmented audio arrays.

    Args:
        audio (np.array): Original audio samples.
        sr (int): Sample rate.
        do_time_shift (bool): If True, apply time shift.
        do_pitch_shift (bool): If True, apply pitch shifting.
        do_speed_change (bool): If True, apply time stretching/compressing.
        do_noise (bool): If True, add colored noise.
        noise_type (str): Type of noise if do_noise=True.
        noise_factor (float): Noise amplitude scaling factor.
        pitch_range (float): Range (±) in semitones for pitch shift.

    Returns:
        List[np.array]: a list of augmented audio arrays.
    """
    augmented_audios = []
    
    original_count = 0  # track how many augmentations succeed

    # 1) Time Shift
    if do_time_shift:
        shifted = time_shift(audio)
        augmented_audios.append(shifted)
        original_count += 1

    # 2) Pitch Shift
    if do_pitch_shift:
        pitched = change_pitch(audio, sr=sr, pitch_range=pitch_range)
        augmented_audios.append(pitched)
        original_count += 1

    # 3) Speed Change
    if do_speed_change:
        sped = change_speed(audio)
        augmented_audios.append(sped)
        original_count += 1

    # 4) Noise Injection
    if do_noise:
        noisy = add_colored_noise(audio, noise_type=noise_type, noise_factor=noise_factor)
        augmented_audios.append(noisy)
        original_count += 1

    logging.info(f"Created {original_count} augmentations for one audio clip.")
    return augmented_audios

def augment_audio_with_repetitions(
    audio, sr,
    # Use constants.py as defaults, so you can override if desired
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

    # NEW: Additional augmentation toggles
    do_compression=AUG_DO_COMPRESSION,
    compression_count=AUG_COMPRESSION_COUNT,

    do_reverb=AUG_DO_REVERB,
    reverb_count=AUG_REVERB_COUNT,

    do_equalize=AUG_DO_EQUALIZE,
    equalize_count=AUG_EQUALIZE_COUNT
):
    """
    Loops multiple times per augmentation type, using default values from constants.py.
    Each call is random, so you get different results for each loop iteration.
    Now includes optional dynamic range compression, reverb, and equalization.
    """
    augmented_audios = []

    # Multiple random time shifts
    if do_time_shift:
        for _ in range(time_shift_count):
            shifted = time_shift(audio, shift_max=shift_max)
            augmented_audios.append(shifted)

    # Multiple random pitch shifts
    if do_pitch_shift:
        for _ in range(pitch_shift_count):
            pitched = change_pitch(audio, sr=sr, pitch_range=pitch_range)
            augmented_audios.append(pitched)

    # Multiple random speed changes
    if do_speed_change:
        for _ in range(speed_change_count):
            sped = change_speed(audio, speed_range=speed_range)
            augmented_audios.append(sped)

    # Multiple random noise injections
    if do_noise:
        for _ in range(noise_count):
            noisy = add_colored_noise(audio, noise_type=noise_type, noise_factor=noise_factor)
            augmented_audios.append(noisy)

    # NEW: Multiple dynamic range compressions
    if do_compression:
        for _ in range(compression_count):
            compressed = dynamic_range_compression(audio)
            augmented_audios.append(compressed)

    # NEW: Multiple reverb passes
    if do_reverb:
        for _ in range(reverb_count):
            reverbed = add_reverb(audio)
            augmented_audios.append(reverbed)

    # NEW: Multiple equalizations
    if do_equalize:
        for _ in range(equalize_count):
            eq_clip = equalize(audio)
            augmented_audios.append(eq_clip)

    logging.info(f"Created {len(augmented_audios)} augmented clips for one original audio.")
    return augmented_audios
