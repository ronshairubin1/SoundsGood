import os

class Config:
    """
    Configuration settings for the application.
    Centralizes all configuration parameters for easy management.
    """
    # App directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    UPLOAD_DIR = os.path.join(BASE_DIR, 'uploads')
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    TEMP_DIR = os.path.join(BASE_DIR, 'temp')
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    DICTIONARIES_DIR = os.path.join(BASE_DIR, 'dictionaries')
    
    # Define specific sound directories within data/sounds
    RAW_SOUNDS_DIR = os.path.join(DATA_DIR, 'sounds', 'raw_sounds')
    PENDING_VERIFICATION_SOUNDS_DIR = os.path.join(DATA_DIR, 'sounds', 'pending_verification_live_recording_sounds')
    UPLOADED_SOUNDS_DIR = os.path.join(DATA_DIR, 'sounds', 'uploaded_sounds')
    TEST_SOUNDS_DIR = os.path.join(DATA_DIR, 'sounds', 'test_sounds')
    TRAINING_SOUNDS_DIR = os.path.join(DATA_DIR, 'sounds', 'training_sounds')
    
    # Ensure all directories exist
    for directory in [UPLOAD_DIR, MODELS_DIR, TEMP_DIR, DATA_DIR, DICTIONARIES_DIR, 
                      TRAINING_SOUNDS_DIR, RAW_SOUNDS_DIR, PENDING_VERIFICATION_SOUNDS_DIR,
                      UPLOADED_SOUNDS_DIR, TEST_SOUNDS_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # Audio processing settings
    SAMPLE_RATE = 16000
    SOUND_THRESHOLD = 0.02
    SILENCE_THRESHOLD = 0.02
    MIN_CHUNK_DURATION = 0.2
    MIN_SILENCE_DURATION = 0.1
    MAX_SILENCE_DURATION = 10.0
    
    # New audio processing parameters
    AMBIENT_NOISE_DURATION = 3.0
    SOUND_MULTIPLIER = 3.0
    SOUND_END_MULTIPLIER = 2.0
    PADDING_DURATION = 0.01
    ENABLE_STRETCHING = False
    TARGET_CHUNK_DURATION = 1.0
    AUTO_STOP_AFTER_SILENCE = 10.0
    
    # ML model settings
    CNN_PARAMS = {
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001,
        'n_mels': 128,
        'n_fft': 2048,
        'hop_length': 512
    }
    
    RF_PARAMS = {
        'n_estimators': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1
    }
    
    ENSEMBLE_PARAMS = {
        'rf_weight': 0.5
    }
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev_key_for_testing')
    DEBUG = os.environ.get('FLASK_DEBUG', 'True').lower() in ['true', '1', 't']
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max upload size
    
    # Logging settings
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    @classmethod
    def init_directories(cls):
        """
        Ensure all required directories exist.
        This is called during application startup.
        """
        directories = [
            cls.UPLOAD_DIR,
            cls.MODELS_DIR,
            cls.TEMP_DIR,
            cls.DATA_DIR,
            cls.DICTIONARIES_DIR,
            cls.TRAINING_SOUNDS_DIR,
            cls.RAW_SOUNDS_DIR,
            cls.PENDING_VERIFICATION_SOUNDS_DIR,
            cls.UPLOADED_SOUNDS_DIR,
            cls.TEST_SOUNDS_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
        return True
    
    @classmethod
    def get_model_path(cls, model_name, model_type):
        """
        Get the full path for a model file based on its name and type.
        
        Args:
            model_name (str): Name of the model
            model_type (str): Type of model ('cnn', 'rf', or 'ensemble')
            
        Returns:
            str: Full path to the model file
        """
        extension = ''
        if model_type.lower() == 'cnn':
            extension = '.h5'
        elif model_type.lower() == 'rf':
            extension = '.joblib'
        
        return os.path.join(cls.MODELS_DIR, f"{model_name}{extension}")
    
    @classmethod
    def get_upload_path(cls, filename):
        """
        Get the full path for an uploaded file.
        
        Args:
            filename (str): Name of the uploaded file
            
        Returns:
            str: Full path to the uploaded file
        """
        return os.path.join(cls.UPLOAD_DIR, filename)
    
    @classmethod
    def get_temp_path(cls, filename):
        """
        Get a temporary file path.
        
        Args:
            filename (str): Name for the temporary file
            
        Returns:
            str: Full path to the temporary file
        """
        return os.path.join(cls.TEMP_DIR, filename) 