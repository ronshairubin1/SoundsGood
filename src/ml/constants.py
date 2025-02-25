# SoundClassifier_v08/src/ml/constants.py

# -----------------------------
# Global settings
# -----------------------------
SAMPLE_RATE = 16000
N_MFCC = 13
AUDIO_DURATION = 1.0  # Use fixed duration of 1 second
AUDIO_LENGTH = int(SAMPLE_RATE * AUDIO_DURATION)
BATCH_SIZE = 32  # Increased batch size
EPOCHS = 50      # Increased epochs 

# -----------------------------
# Test settings
# -----------------------------
TEST_DURATION = 5  # seconds
TEST_FS = 44100  # Sample rate

# -----------------------------
# AUDIO SCALING CONSTANTS
# -----------------------------
INT16_MAX = 32767  # Maximum value for int16

# -----------------------------
# AUGMENTATION DEFAULTS
# -----------------------------
# You can modify these defaults in one place for all models:
AUG_DO_TIME_SHIFT = True
AUG_TIME_SHIFT_COUNT = 3
AUG_SHIFT_MAX = 0.2

AUG_DO_PITCH_SHIFT = True
AUG_PITCH_SHIFT_COUNT = 3
AUG_PITCH_RANGE = 2.0

AUG_DO_SPEED_CHANGE = True
AUG_SPEED_CHANGE_COUNT = 3
AUG_SPEED_RANGE = 0.1

AUG_DO_NOISE = True
AUG_NOISE_COUNT = 3
AUG_NOISE_TYPE = "white"
AUG_NOISE_FACTOR = 0.005

# -----------------------------
# ADDITIONAL AUGMENTATION CONSTANTS
# -----------------------------

# For advanced pitch shifting arrays:
PITCH_SHIFTS_OUTER_VALUES = [-3.0, -2.0, 2.0, 3.0]
PITCH_SHIFTS_CENTER_START = -1.0
PITCH_SHIFTS_CENTER_END = 1.0
PITCH_SHIFTS_CENTER_NUM = 9

# For advanced noise experimentation:
NOISE_TYPES_LIST = ['white', 'pink', 'brown']
NOISE_LEVELS_MIN = 0.001
NOISE_LEVELS_MAX = 0.01
NOISE_LEVELS_COUNT = 5

# ---------------------------------
# NEW: Additional effect toggles

# ---------------------------------
AUG_DO_COMPRESSION = False
AUG_COMPRESSION_COUNT = 1

AUG_DO_REVERB = True
AUG_REVERB_COUNT = 1

AUG_DO_EQUALIZE = True
AUG_EQUALIZE_COUNT = 1