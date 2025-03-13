import os
import json
import logging
import uuid
import shutil
import hashlib
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_from_directory
from datetime import datetime
from flask_cors import CORS
from backend.config import Config
from backend.src.ml.model_paths import get_model_counts_from_registry, synchronize_model_registry, count_model_files_from_registry

# Configure logging
logging.basicConfig(
    filename=os.path.join(Config.LOGS_DIR, 'app_execution.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('main')
logger.info("=" * 50)
logger.info("Starting Sound Classifier app with new architecture...")

logger.info("Loaded Config from config.py")

# Import our new API instead of the old routes
try:
    from backend.src.api.ml_api import MlApi
    logger.info("Imported MlApi from backend.src.api.ml_api")
except ImportError as e:
    logger.error("Failed to import MlApi: %s", e)
    import sys
    logger.critical("MlApi is a critical component. Exiting application due to import failure.")
    sys.exit(1)

try:
    from backend.src.api.dictionary_api import dictionary_bp
    logger.info("Imported dictionary_bp from backend.src.api.dictionary_api")
except ImportError as e:
    logger.error("Failed to import dictionary_bp: %s", e)

try:
    from backend.src.api.user_api import user_bp
    logger.info("Imported user_bp from backend.src.api.user_api")
except ImportError as e:
    logger.error("Failed to import user_bp: %s", e)

try:
    from backend.src.api.dashboard_api import dashboard_bp
    logger.info("Imported dashboard_bp from backend.src.api.dashboard_api")
except ImportError as e:
    logger.error("Failed to import dashboard_bp: %s", e)

# Import the ML routes
try:
    # Import the ml_bp blueprint
    from backend.src.routes.ml_routes import ml_bp
    logger.info("Imported ml_bp from backend.src.routes.ml_routes")
except ImportError as e:
    logger.error("Failed to import ml_bp: %s", e)
    # Fallback: Try to import from ml_api if ml_routes fails
    try:
        from backend.src.api.ml_api import ml_bp
        logger.info("Fallback: Imported ml_bp from backend.src.api.ml_api instead")
    except ImportError as e2:
        logger.error("Failed fallback import of ml_bp from ml_api: %s", e2)

# Import the recording_bp blueprint for our new audio processing features
try:
    from backend.src.routes.recording_routes import recording_bp
    logger.info("Imported recording_bp from backend.src.routes.recording_routes")
except ImportError as e:
    logger.error("Failed to import recording_bp: %s", e)

# Import services
try:
    from backend.src.services.dictionary_service import DictionaryService
    logger.info("Imported DictionaryService from backend.src.services.dictionary_service")
except ImportError as e:
    logger.error("Failed to import DictionaryService: %s", e)

try:
    from backend.src.services.user_service import UserService
    logger.info("Imported UserService from backend.src.services.user_service")
except ImportError as e:
    logger.error("Failed to import UserService: %s", e)

# --------------------------------------------------------------------
# Set up Flask
# --------------------------------------------------------------------
# Use absolute paths for template and static folders
template_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend', 'templates')
static_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend', 'static')

app = Flask(
    __name__,
    static_url_path='/static',
    static_folder=static_folder,
    template_folder=template_folder
)
logger.debug("Template folder: %s", template_folder)
logger.debug("Static folder: %s", static_folder)

app.secret_key = Config.SECRET_KEY
CORS(app, supports_credentials=True)

# The test login route is defined at the bottom of this file

# Ensure required directories exist
os.makedirs(Config.TEMP_DIR, exist_ok=True)
logger.debug("Ensured TEMP_DIR exists: %s", Config.TEMP_DIR)
os.makedirs(Config.TRAINING_SOUNDS_DIR, exist_ok=True)
logger.debug("Ensured TRAINING_SOUNDS_DIR exists: %s", Config.TRAINING_SOUNDS_DIR)

# Register blueprints
try:
    app.register_blueprint(dictionary_bp)
    logger.info("Registered dictionary_bp blueprint")
    app.register_blueprint(user_bp)
    logger.info("Registered user_bp blueprint")
    app.register_blueprint(dashboard_bp)
    logger.info("Registered dashboard_bp blueprint")
    # Register the ml_bp blueprint
    try:
        app.register_blueprint(ml_bp)
        logger.info("Registered ml_bp blueprint")
    except NameError as e:
        logger.error(f"Failed to register ml_bp blueprint: {e}")
    app.register_blueprint(recording_bp, url_prefix='/recording')  # Register our new recording_bp
    logger.info("Registered recording_bp blueprint")
except (ImportError, AttributeError) as e:
    logger.error("Error registering blueprints: %s", e)

# Initialize services
dictionary_service = DictionaryService()
logger.info("Initialized DictionaryService")
user_service = UserService()
logger.info("Initialized UserService")

# Initialize the ML API with our Flask app
ml_api = MlApi(app, model_dir=Config.MODELS_DIR)

# Add direct routes to support frontend
@app.route('/model_metadata/<model_id>', methods=['GET'])
def model_metadata_direct(model_id):
    """Direct route for model metadata to support frontend without blueprint prefix"""
    from backend.src.routes.ml_routes import get_model_metadata_direct
    return get_model_metadata_direct(model_id)

# --------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------
# Configure app logger with a file handler
file_handler = logging.FileHandler(os.path.join(Config.LOGS_DIR, 'flask_app.log'))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(logging.Formatter(Config.LOG_FORMAT))
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.DEBUG)
app.logger.debug("Starting Sound Classifier app with new architecture...")
app.logger.debug("Template folder: %s", app.template_folder)
app.logger.debug("Static folder: %s", app.static_folder)

# --------------------------------------------------------------------
# Initialize directories
# --------------------------------------------------------------------
Config.init_directories()

# Initialize application directories
def init_app_directories():
    """
    Initialize required application directories
    """
    # Create the sounds directory if it doesn't exist
    if not os.path.exists(Config.TRAINING_SOUNDS_DIR):
        os.makedirs(Config.TRAINING_SOUNDS_DIR, exist_ok=True)
        app.logger.info("Created sounds directory: %s", Config.TRAINING_SOUNDS_DIR)
    
    # Create the temp directory if it doesn't exist
    if not os.path.exists(Config.TEMP_DIR):
        os.makedirs(Config.TEMP_DIR, exist_ok=True)
        app.logger.info("Created temp directory: %s", Config.TEMP_DIR)

# Initialize application directories
init_app_directories()

# --------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------
def get_all_sound_classes():
    """
    Centralized function to get all available sound classes.
    This serves as the single source of truth for sound classes in the application.
    
    Returns:
        list: A list of all sound class names
    """
    # Get classes from the training sounds directory (these may or may not have sounds)
    training_classes = set()
    if os.path.exists(Config.TRAINING_SOUNDS_DIR):
        training_classes = {d for d in os.listdir(Config.TRAINING_SOUNDS_DIR)
                          if os.path.isdir(os.path.join(Config.TRAINING_SOUNDS_DIR, d))}
    
    # Get classes from dictionaries (these may or may not have sounds)
    dictionary_classes = set()
    try:
        for dict_info in dictionary_service.get_dictionaries() or []:
            if 'classes' in dict_info and dict_info['classes']:
                dictionary_classes.update(dict_info['classes'])
            elif 'sounds' in dict_info and dict_info['sounds']:
                dictionary_classes.update(dict_info['sounds'])
    except Exception as e:
        app.logger.error("Error getting classes from dictionaries: %s", str(e))
    
    # Combine both sources
    all_classes = training_classes.union(dictionary_classes)
    
    # Convert to sorted list
    return sorted(list(all_classes))

def get_classes_with_sounds():
    """
    Get all sound classes that have at least one sound recording.
    This distinguishes between empty and non-empty classes.
    
    Returns:
        dict: A dictionary where keys are class names and values are the count of recordings
    """
    classes_with_sounds = {}
    all_classes = get_all_sound_classes()
    
    # Check which classes have sound recordings
    if os.path.exists(Config.TRAINING_SOUNDS_DIR):
        for class_name in all_classes:
            class_dir = os.path.join(Config.TRAINING_SOUNDS_DIR, class_name)
            if os.path.exists(class_dir) and os.path.isdir(class_dir):
                sound_files = [f for f in os.listdir(class_dir) 
                             if f.endswith('.wav') and os.path.isfile(os.path.join(class_dir, f))]
                if sound_files:
                    classes_with_sounds[class_name] = len(sound_files)
    
    return classes_with_sounds

def get_augmented_recordings_from_json():
    """
    Get the count of augmented recordings directly from classes.json
    This provides a single source of truth for augmented recording counts.
    
    Returns:
        int: The total count of augmented recordings across all classes
    """
    try:
        # The correct path to classes.json is in backend/data/classes/classes.json
        classes_json_path = os.path.join(Config.BACKEND_DATA_DIR, 'classes', 'classes.json')
        app.logger.info(f"Looking for classes.json at {classes_json_path}")
        
        if os.path.exists(classes_json_path):
            with open(classes_json_path, 'r') as f:
                classes_data = json.load(f)
                
            # Sum up all the augmented counts
            augmented_count = 0
            for class_info in classes_data.get('classes', {}).values():
                augmented_count += class_info.get('samples', {}).get('augmented', 0)
                
            app.logger.info(f"Read augmented count from classes.json: {augmented_count}")
            return augmented_count
        app.logger.warning(f"classes.json not found at {classes_json_path}")
        return 0
    except Exception as e:
        app.logger.error(f"Error getting augmented count from JSON: {str(e)}")
        return 0

def get_detailed_sound_classes_info():
    """
    Get detailed information about sound classes including counts of original vs. augmented recordings.
    Each original and augmented recording has an associated dataset from feature extraction.
    
    Returns:
        dict: A dictionary with class information including counts of original and augmented recordings
              Format: {class_name: {'original': count, 'augmented': count, 'total': count}}
    """
    try:
        detailed_classes = {}
        all_classes = get_all_sound_classes()
        
        # Check which classes have sound recordings and count original vs augmented
        if os.path.exists(Config.TRAINING_SOUNDS_DIR):
            for class_name in all_classes:
                class_dir = os.path.join(Config.TRAINING_SOUNDS_DIR, class_name)
                augmented_dir = os.path.join(Config.AUGMENTED_SOUNDS_DIR, class_name)  # Single source of truth for augmented files
                
                if os.path.exists(class_dir) and os.path.isdir(class_dir):
                    try:
                        # Get original audio files from the training directory
                        original_recordings = [f for f in os.listdir(class_dir)
                                        if (f.endswith('.wav') or f.endswith('.mp3')) and
                                        os.path.isfile(os.path.join(class_dir, f))]
                        
                        # Get augmented audio files from the primary augmented directory (single source of truth)
                        augmented_recordings = []
                        
                        # Check augmented directory (/backend/data/sounds/augmented/{class})
                        if os.path.exists(augmented_dir) and os.path.isdir(augmented_dir):
                            augmented_recordings = [f for f in os.listdir(augmented_dir)
                                            if (f.endswith('.wav') or f.endswith('.mp3')) and
                                            os.path.isfile(os.path.join(augmented_dir, f)) and
                                            ('_aug_' in f or ('_p' in f and '_t' in f and '_n' in f))]
                        
                        original_count = len(original_recordings)
                        augmented_count = len(augmented_recordings)
                        total_count = original_count + augmented_count
                        
                        if original_count > 0 or augmented_count > 0:  # Only include classes with at least one recording
                            detailed_classes[class_name] = {
                                'original': original_count,
                                'augmented': augmented_count,
                                'total': total_count
                            }
                            app.logger.debug("Class %s: %d original, %d augmented", 
                                               class_name, original_count, augmented_count)
                    except Exception as e:
                        app.logger.error("Error processing class directory %s: %s", class_dir, str(e))
                        # Add an empty entry for this class to maintain consistency
                        detailed_classes[class_name] = {
                            'original': 0,
                            'augmented': 0,
                            'total': 0
                        }
        
        return detailed_classes
    except Exception as e:
        app.logger.error("Error in get_detailed_sound_classes_info: %s", str(e))
        # Return an empty dictionary in case of errors
        return {}

# --------------------------------------------------------------------
# Basic Routes: index, login, logout, register
# --------------------------------------------------------------------
@app.route('/')
def index():
    # Debug the session state
    app.logger.info("INDEX ROUTE: Session contains: %s", session)
    
    # If not logged in, ask for login
    if 'username' not in session:
        app.logger.warning("Username not found in session, redirecting to login")
        return render_template('login.html')
    
    # If logged in, go to dashboard with stats
    # Prepare dashboard stats
    stats = {
        'dictionaries': 0,
        'classes': 0,
        'original_recordings': 0,
        'recordings': 0,
        'augmented_recordings': 0,
        'total_recordings': 0,
        'models': 0
    }
    
    # Get statistics from services if possible
    try:
        # Count dictionaries
        dictionaries = dictionary_service.get_dictionaries(session.get('user_id'))
        stats['dictionaries'] = len(dictionaries) if dictionaries else 0
        
        # Get all sound classes (empty or non-empty) using the centralized function
        all_classes = get_all_sound_classes()
        stats['classes'] = len(all_classes)
        
        # Get classes that have at least one sound recording
        classes_with_sounds = get_classes_with_sounds()
        
        try:
            # Get detailed sound class information including counts of original and augmented recordings
            detailed_classes = get_detailed_sound_classes_info()
            
            # Count original and augmented recordings from the detailed class info
            original_recordings = 0
            augmented_recordings = 0
            
            # Calculate totals from the detailed class information
            for class_info in detailed_classes.values():
                original_recordings += class_info['original']
                augmented_recordings += class_info['augmented']
            
            # Set original recordings stats
            stats['original_recordings'] = original_recordings
            stats['recordings'] = original_recordings  # Also set as recordings for backward compatibility
            
            # Get augmented recordings count directly from classes.json
            stats['augmented_recordings'] = get_augmented_recordings_from_json()
            stats['total_recordings'] = original_recordings + stats['augmented_recordings']
        except Exception as e:
            app.logger.error("Error calculating recording stats: %s", str(e))
            # Fall back to a simpler method if the detailed method fails
            try:
                # Count recordings using the traditional method
                original_recordings = 0
                augmented_recordings = 0
                
                # Loop through all classes with sounds
                for class_name in classes_with_sounds:
                    class_path = os.path.join(Config.TRAINING_SOUNDS_DIR, class_name)
                    
                    # Get all audio files
                    all_recordings = [f for f in os.listdir(class_path) if f.endswith('.wav') or f.endswith('.mp3')]
                    
                    # Count augmented recordings using the same criteria as recording_service.py
                    aug_files = [f for f in all_recordings if 
                               ('_p' in f and '_t' in f and '_n' in f) or  
                               'aug' in f.lower()]
                    augmented_recordings += len(aug_files)
                    
                    # Original recordings are all files minus augmented ones
                    original_recordings += (len(all_recordings) - len(aug_files))
                
                # Set original recordings stats
                stats['recordings'] = original_recordings
                
                # Get augmented recordings count directly from classes.json
                stats['augmented_recordings'] = get_augmented_recordings_from_json()
                stats['total_recordings'] = original_recordings + stats['augmented_recordings']
            except Exception as e:
                app.logger.error("Error in fallback recording stats calculation: %s", str(e))
                # If all else fails, set some safe default values
                stats['recordings'] = 0
                stats['augmented_recordings'] = 0
                stats['total_recordings'] = 0
        
        app.logger.info("Dashboard stats: %s", stats)
        
        # Count all models from the registry
        model_count, model_types_count = count_model_files_from_registry()
        app.logger.info("Found %d total model files", model_count)
        app.logger.info("Model type breakdown - CNN: %d, RF: %d, Ensemble: %d", 
                      model_types_count['cnn'], model_types_count['rf'], model_types_count['ensemble'])
        
        stats['models'] = model_count
        stats['model_types'] = model_types_count  # Add detailed model type counts to stats
        app.logger.info("Total model count: %d", model_count)
        
        # Override stats with known correct values to ensure dashboard shows accurate data
        # These values come from the detailed class information processing
        
        # Make sure we set the original_recordings field
        stats['original_recordings'] = stats.get('recordings', 0)
        
        # Make sure we have the augmented_recordings field using the direct method from classes.json
        # This ensures we have the most accurate count regardless of other calculations
        stats['augmented_recordings'] = get_augmented_recordings_from_json()
        stats['total_recordings'] = stats['original_recordings'] + stats['augmented_recordings']
        
        # Print very clear debug information about the augmented recordings count
        app.logger.info("======= AUGMENTED RECORDINGS COUNT: %s =======", stats['augmented_recordings'])
        app.logger.info("Dashboard stats after override: %s", stats)
        
        # Mock data for recent activities
        recent_activities = [
            {
                'type': 'record',
                'icon': 'mic-fill',
                'description': 'Recorded a new sound sample',
                'time': '5 minutes ago'
            },
            {
                'type': 'train',
                'icon': 'gear-fill',
                'description': 'Trained a new model',
                'time': '2 hours ago'
            },
            {
                'type': 'predict',
                'icon': 'soundwave',
                'description': 'Classified a sound with 95% accuracy',
                'time': 'Yesterday'
            }
        ]
        
        return render_template('dashboard.html', 
                              stats=stats,
                              recent_activities=recent_activities)
    except Exception as e:
        app.logger.error(f"Error generating dashboard: {str(e)}")
        return render_template('error.html', error="An error occurred loading the dashboard. Please try again.")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if not username or not password:
            flash('Please provide both username and password', 'danger')
            return render_template('login.html')
        
        # Use the user service to authenticate
        result = user_service.login(username, password)
        
        if result['success']:
            # Set session data
            session['username'] = username
            session['user_id'] = result['user']['id']
            session['is_admin'] = result['user']['is_admin']
            flash('Welcome back, {}!'.format(username), 'success')
            return redirect(url_for('index'))
        
        flash(result['error'], 'danger')
        return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/test-login')
def test_login():
    """
    Test route to display login form with Ron's credentials pre-filled
    This is for development/debugging only
    """
    # Redirect to login page with Ron's credentials pre-filled
    flash('Please login with your account (username and password are pre-filled)', 'info')
    return render_template('login.html', ron_mode=True)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')

    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')
        
        if not username or not password:
            flash('Please provide both username and password', 'danger')
            return render_template('register.html')
        
        # Use the user service to register
        result = user_service.register(username, password, email)
        
        if result['success']:
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        
        flash(result['error'], 'danger')
        return render_template('register.html')

# --------------------------------------------------------------------
# Dictionary Management Routes
# --------------------------------------------------------------------
@app.route('/dictionaries')
def list_dictionaries():
    """Render the dictionary management page."""
    app.logger.info("=== MANAGE DICTIONARIES PAGE REQUESTED ===")
    
    if 'username' not in session:
        app.logger.warning("No username in session, redirecting to login")
        return redirect(url_for('login'))
    
    # Get dictionaries
    app.logger.info("Fetching dictionaries for user: %s", session.get('user_id'))
    
    # Explicitly reload the dictionary service
    app.logger.info("Force reloading dictionary service data")
    dictionary_service._load_dictionaries()
    
    dictionaries = dictionary_service.get_dictionaries(session.get('user_id'))
    app.logger.info("Found %d dictionaries: %s", len(dictionaries), [d.get('name') for d in dictionaries])
    
    return render_template('dictionaries.html', dictionaries=dictionaries)

@app.route('/dictionaries/<dict_name>')
def manage_dictionary(dict_name):
    """Render the dictionary management page."""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Force reload of dictionaries to ensure we have the latest data
    dictionary_service._load_dictionaries()
    
    # Get dictionary
    dictionary = dictionary_service.get_dictionary(dict_name)
    
    if not dictionary:
        flash(f'Dictionary "{dict_name}" not found', 'danger')
        return redirect(url_for('list_dictionaries'))
    
    # Log classes for debugging
    if dictionary.get('classes'):
        app.logger.debug("Dictionary classes: %s", dictionary['classes'])
    else:
        app.logger.debug("Dictionary has no classes defined")
    
    return render_template('dictionary_manager.html', dictionary=dictionary)

@app.route('/dictionaries/<dict_name>/export')
def export_dictionary(dict_name):
    """Export a dictionary as JSON."""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get dictionary
    dictionary = dictionary_service.get_dictionary(dict_name)
    
    if not dictionary:
        flash(f'Dictionary "{dict_name}" not found', 'danger')
        return redirect(url_for('list_dictionaries'))
    
    # Create a response with the dictionary data
    response = jsonify(dictionary)
    response.headers.set('Content-Disposition', 'attachment; filename={}_export.json'.format(dict_name.replace(" ", "_").lower()))
    return response

# --------------------------------------------------------------------
# Training and Prediction Routes
# --------------------------------------------------------------------
@app.route('/training')
def training_redirect():
    """Redirect to the ML training blueprint route."""
    return redirect(url_for('ml.training_view'))

@app.route('/train')
def train_redirect():
    """Redirect to the ML training blueprint route, now pointing to the advanced interface."""
    return redirect(url_for('ml.training_view'))

@app.route('/predict')
def predict_view():
    """Render the prediction page."""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    return render_template('predict.html')

# --------------------------------------------------------------------
# Sound Management Routes
# --------------------------------------------------------------------
@app.route('/sounds/manage')
def manage_sounds():
    """Render the sounds management page."""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get all sound classes using our centralized function
    all_classes = get_all_sound_classes()
    
    # Get classes with sounds and their counts
    classes_with_sounds = get_classes_with_sounds()
    
    sound_classes = []
    
    # Process all classes, including those without sounds
    for class_name in all_classes:
        class_path = os.path.join(Config.TRAINING_SOUNDS_DIR, class_name)
        
        # Create the directory if it doesn't exist yet (ensuring all possible classes are available)
        if not os.path.exists(class_path):
            os.makedirs(class_path, exist_ok=True)
        
        # Get sample count from our mapping or default to 0 for empty classes
        sample_count = classes_with_sounds.get(class_name, 0)
        
        sound_classes.append({
            'name': class_name,
            'sample_count': sample_count,
            'path': class_path
        })
    
    # Sort sound classes alphabetically by name
    sound_classes.sort(key=lambda x: x['name'].lower())
    
    return render_template('sounds_management.html', sound_classes=sound_classes)

@app.route('/sounds/record')
def record_sounds():
    """Render the sounds recording page."""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Use our centralized function to get all sound classes
    all_classes = get_all_sound_classes()
    sound_classes = []
    
    try:
        # Get classes with sounds using our centralized function
        classes_with_sounds = get_classes_with_sounds()
        
        # Process all classes, including those without sounds
        for class_name in all_classes:
            class_path = os.path.join(Config.TRAINING_SOUNDS_DIR, class_name)
            
            # Create directory if it doesn't exist (to ensure all classes from dictionaries are available)
            if not os.path.exists(class_path):
                os.makedirs(class_path, exist_ok=True)
            
            # Get sample count from our mapping or default to 0 for empty classes
            sample_count = classes_with_sounds.get(class_name, 0)
            
            sound_classes.append({
                'name': class_name,
                'sample_count': sample_count,
                'path': class_path
            })
    except Exception as e:
        app.logger.error("Error getting class information: %s", str(e))
        
        # Fall back to direct directory reading if centralized function fails
        for class_name in all_classes:
            class_path = os.path.join(Config.TRAINING_SOUNDS_DIR, class_name)
            
            # Create directory if it doesn't exist (to ensure all classes from dictionaries are available)
            if not os.path.exists(class_path):
                os.makedirs(class_path, exist_ok=True)
                samples = []
            else:
                samples = [f for f in os.listdir(class_path) if f.lower().endswith('.wav')]
            
            sound_classes.append({
                'name': class_name,
                'sample_count': len(samples),
                'path': class_path
            })
    
    # Sort sound classes alphabetically by name
    sound_classes.sort(key=lambda x: x['name'].lower())
    
    # Check if we're coming from a dictionary page
    source_dict_name = request.args.get('dict_name')
    source_dictionary = None
    dict_classes = []
    
    if source_dict_name:
        # Get dictionary information
        source_dictionary = dictionary_service.get_dictionary(source_dict_name)
        
        if source_dictionary and 'classes' in source_dictionary:
            # Get classes from this dictionary
            dict_classes = [cls for cls in sound_classes if cls['name'] in source_dictionary['classes']]
    
    return render_template('sounds_record.html', 
                          sound_classes=sound_classes,
                          source_dictionary=source_dictionary,
                          dict_classes=dict_classes,
                          use_unified_processor=True)  # Flag to use the new unified processor

@app.route('/sounds/class/<class_name>')
def view_sound_class(class_name):
    """Render the sound class view page."""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get samples for the class
    class_path = os.path.join(Config.TRAINING_SOUNDS_DIR, class_name)
    
    if not os.path.exists(class_path):
        flash(f'Sound class "{class_name}" not found', 'danger')
        return redirect(url_for('manage_sounds'))
    
    # Get samples
    samples = []
    for sample_file in os.listdir(class_path):
        if sample_file.lower().endswith('.wav'):
            sample_path = os.path.join(class_path, sample_file)
            samples.append({
                'name': sample_file,
                'path': f'/sounds/{class_name}/{sample_file}',
                'size': os.path.getsize(sample_path)
            })
    
    # Sort samples by name
    samples.sort(key=lambda x: x['name'])
    
    return render_template('sound_class_view.html', 
                          class_name=class_name,
                          samples=samples)

# --------------------------------------------------------------------
# Sound Management API
# --------------------------------------------------------------------
@app.route('/api/sounds/classes', methods=['GET'])
def api_get_sound_classes():
    """API endpoint for getting sound classes with detailed information about original and augmented recordings."""
    try:
        # Get detailed information about sound classes
        detailed_classes = get_detailed_sound_classes_info()
        
        result = []
        for class_name, class_info in sorted(detailed_classes.items()):
            result.append({
                'name': class_name,
                'sample_count': class_info['total'],
                'original_count': class_info['original'],
                'augmented_count': class_info['augmented']
            })
        
        return jsonify({
            'success': True,
            'classes': result
        })
    except Exception as e:
        app.logger.error(f"Error in api_get_sound_classes: {str(e)}")
        # Return a basic response in case of errors
        training_sounds_dir = Config.TRAINING_SOUNDS_DIR
        result = []
        
        if os.path.exists(training_sounds_dir):
            try:
                class_dirs = [d for d in os.listdir(training_sounds_dir)
                             if os.path.isdir(os.path.join(training_sounds_dir, d))]
                
                for class_name in sorted(class_dirs):
                    class_path = os.path.join(training_sounds_dir, class_name)
                    try:
                        samples = [f for f in os.listdir(class_path) if f.lower().endswith('.wav')]
                        
                        result.append({
                            'name': class_name,
                            'sample_count': len(samples),
                            'original_count': 0,  # Fallback values
                            'augmented_count': 0   # Fallback values
                        })
                    except Exception:
                        # If we can't read a specific class directory, just skip it
                        pass
            except Exception:
                # If we can't read the training sounds directory, return an empty list
                pass
        
        return jsonify({
            'success': True,
            'classes': result
        })

@app.route('/api/sounds/classes', methods=['POST'])
def api_create_sound_class():
    """Create a new sound class."""
    data = request.get_json()
    
    if not data or 'class_name' not in data:
        return jsonify({
            'success': False,
            'error': 'Class name is required'
        }), 400
    
    class_name = data['class_name']
    safe_class_name = class_name.replace(' ', '_').lower()
    
    # Create class directory in training sounds directory
    class_path = os.path.join(Config.TRAINING_SOUNDS_DIR, safe_class_name)
    
    if os.path.exists(class_path):
        return jsonify({
            'success': False,
            'error': 'Class "{}" already exists'.format(class_name)
        }), 400
    
    try:
        # Create directory in main sounds folder
        os.makedirs(class_path, exist_ok=True)
        app.logger.info("Created class directory in TRAINING_SOUNDS_DIR: %s", class_path)
        
        # Create directories in data/sounds subdirectories
        # 1. Raw sounds directory
        raw_class_path = os.path.join(Config.RAW_SOUNDS_DIR, safe_class_name)
        os.makedirs(raw_class_path, exist_ok=True)
        app.logger.info("Created class directory in RAW_SOUNDS_DIR: %s", raw_class_path)
        
        # 2. Temp sounds directory
        temp_class_path = os.path.join(Config.PENDING_VERIFICATION_SOUNDS_DIR, safe_class_name)
        os.makedirs(temp_class_path, exist_ok=True)
        app.logger.info("Created class directory in pending verification sounds dir: %s", temp_class_path)
        
        # 3. Training sounds directory
        training_class_path = os.path.join(Config.TRAINING_SOUNDS_DIR, safe_class_name)
        os.makedirs(training_class_path, exist_ok=True)
        app.logger.info("Created class directory in TRAINING_SOUNDS_DIR: %s", training_class_path)
        
        
        return jsonify({
            'success': True,
            'class': {
                'name': safe_class_name,
                'sample_count': 0
            }
        })
    except Exception as e:
        app.logger.error(f"Error creating sound class: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/sounds/classes/<class_name>', methods=['DELETE'])
def api_delete_sound_class(class_name):
    """Delete a sound class and all its recordings."""
    class_path = os.path.join(Config.TRAINING_SOUNDS_DIR, class_name)
    
    if not os.path.exists(class_path):
        return jsonify({
            'success': False,
            'error': 'Class "{}" not found'.format(class_name)
        }), 404
    
    try:
        # Recursively delete the class directory and all its contents
        shutil.rmtree(class_path)
        
        # Remove references to this class from all dictionaries
        try:
            # Update all dictionaries that reference this class
            for dict_name, dict_info in dictionary_service.metadata["dictionaries"].items():
                if class_name in dict_info["classes"]:
                    # Count how many samples were in this class
                    sample_count_reduction = dict_info["sample_count"] // len(dict_info["classes"]) if len(dict_info["classes"]) > 0 else 0
                    
                    # Remove class from dictionary
                    dict_info["classes"].remove(class_name)
                    dict_info["updated_at"] = datetime.now().isoformat()
                    dict_info["sample_count"] = max(0, dict_info["sample_count"] - sample_count_reduction)
            
            # Save updated metadata
            dictionary_service._save_metadata()
        except Exception as dict_error:
            app.logger.error("Error updating dictionaries after class deletion: %s", str(dict_error))
        
        return jsonify({
            'success': True,
            'message': 'Class "{}" and all its recordings deleted successfully'.format(class_name)
        })
    except Exception as e:
        app.logger.error("Error deleting sound class: %s", str(e))
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/sounds/classes/<class_name>/samples', methods=['GET'])
def api_get_class_samples(class_name):
    """Get all samples for a sound class."""
    class_path = os.path.join(Config.TRAINING_SOUNDS_DIR, class_name)
    
    if not os.path.exists(class_path):
        return jsonify({
            'success': False,
            'error': 'Class "{}" not found'.format(class_name)
        }), 404
    
    samples = []
    for sample_file in os.listdir(class_path):
        if sample_file.lower().endswith('.wav'):
            sample_path = os.path.join(class_path, sample_file)
            samples.append({
                'name': sample_file,
                'path': f'/sounds/{class_name}/{sample_file}',
                'size': os.path.getsize(sample_path)
            })
    
    # Sort samples by name
    samples.sort(key=lambda x: x['name'])
    
    return jsonify({
        'success': True,
        'samples': samples
    })

@app.route('/api/sounds/classes/<class_name>/samples/<sample_name>', methods=['DELETE'])
def api_delete_sample(class_name, sample_name):
    """
    API endpoint to delete a sound sample
    """
    try:
        # Check if user is authenticated
        if 'username' not in session:
            return jsonify({'success': False, 'message': 'Authentication required'}), 401
            
        sample_path = os.path.join(Config.TRAINING_SOUNDS_DIR, class_name, sample_name)
        
        if not os.path.exists(sample_path):
            return jsonify({'success': False, 'message': 'Sample not found'}), 404
            
        # Delete the file
        os.remove(sample_path)
        
        # Return success
        return jsonify({'success': True})
    except Exception as e:
        app.logger.error(f"Error in api_delete_sample: {str(e)}")
        return jsonify({
            'success': False, 
            'error': str(e)
        }), 500
        
@app.route('/api/sounds/pending/<class_name>', methods=['GET'])
def api_pending_recordings(class_name):
    """
    API endpoint to get pending recordings for a class
    """
    try:
        app.logger.info(f"Received request for pending recordings for class '{class_name}'")
        
        # Check if user is authenticated
        if 'username' not in session:
            app.logger.warning(f"Authentication required for api_pending_recordings")
            return jsonify({
                'success': False, 
                'message': 'Authentication required'
            }), 401
            
        # Check if there are any pending recordings in the temp_sounds directory
        recordings = []
        
        # Search for pending verification recordings
        pending_sounds_dir = os.path.join(Config.PENDING_VERIFICATION_SOUNDS_DIR, class_name)
        if os.path.exists(pending_sounds_dir):
            app.logger.debug(f"Searching for pending recordings in {pending_sounds_dir}")
            for filename in os.listdir(pending_sounds_dir):
                if filename.endswith('.wav'):
                    # Format file path for the API response
                    file_path = f"{class_name}/{filename}"
                    
                    recordings.append({
                        'filename': file_path,
                        'class_name': class_name,
                        'url': url_for('serve_temp_sound_file', class_name=class_name, filename=filename)
                    })
                    app.logger.debug(f"Found pending recording: {file_path}")
        
        app.logger.info(f"Found {len(recordings)} pending recordings for class '{class_name}'")
        return jsonify({'success': True, 'recordings': recordings})
    except Exception as e:
        app.logger.error(f"Error in api_pending_recordings: {str(e)}")
        return jsonify({'success': False, 'error': str(e), 'recordings': []})

@app.route('/api/sounds/verify', methods=['POST'])
def api_verify_sample():
    """
    API endpoint for verifying sound samples (keep or discard)
    """
    try:
        data = request.json
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})
        
        # Get the filename and keep status
        filename = data.get('filename')
        keep = data.get('keep', False)
        
        if not filename:
            return jsonify({'success': False, 'error': 'No filename provided'})
        
        app.logger.info(f"Processing verification for: {filename}, keep: {keep}")
        
        # The format should now be class_name/filename.wav
        parts = filename.split('/', 1)
        if len(parts) != 2:
            return jsonify({'success': False, 'error': f'Invalid filename format: {filename}'})
        
        class_name = parts[0]
        filename_only = parts[1]
        
        # Move from the pending verification sounds directory to the training sounds directory
        source_path = os.path.join(Config.PENDING_VERIFICATION_SOUNDS_DIR, class_name, filename_only)
        
        if not os.path.exists(source_path):
            return jsonify({'success': False, 'error': f'File not found: {source_path}'})
        
        if keep:
            # Create the training sounds directory if it doesn't exist
            training_dir = os.path.join(Config.TRAINING_SOUNDS_DIR, class_name)
            os.makedirs(training_dir, exist_ok=True)
            
            # Generate a unique filename
            username = session.get('username', 'anonymous')
            timestamp_now = datetime.now().strftime('%Y%m%d%H%M%S')
            new_filename = f"{class_name}_{username}_{timestamp_now}_{uuid.uuid4().hex[:6]}.wav"
            
            # Save to training sounds directory only (no need for redundant checks)
            training_path = os.path.join(training_dir, new_filename)
            shutil.copy2(source_path, training_path)
            app.logger.info(f"Saved approved chunk to {training_path}")
        
        # Delete the temporary file
        try:
            os.remove(source_path)
            app.logger.info(f"Removed source file from pending verification sounds: {source_path}")
        except Exception as del_err:
            app.logger.warning(f"Could not remove source file: {del_err}")
        
        return jsonify({'success': True})
    except Exception as e:
        app.logger.error(f"Error in verify_sample: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/ml/record', methods=['POST'])
def api_ml_record():
    """
    API endpoint for processing sound recordings through the ML pipeline.
    
    This endpoint handles:
    1. Receiving audio data from the client
    2. Processing it through the sound detection pipeline
    3. Saving detected segments for verification
    
    Returns:
        JSON response with success/error information or redirects to verification page
    """
    try:
        # Check if user is authenticated
        if 'username' not in session:
            return jsonify({'success': False, 'message': 'Authentication required'})
        
        # Get form data
        audio_data = request.files.get('audio')
        sound_class = request.form.get('sound')
        measure_ambient = request.form.get('measure_ambient') == 'true'
        use_unified = request.form.get('use_unified') == 'true'
        
        if not audio_data or not sound_class:
            return jsonify({'success': False, 'message': 'Missing required parameters'})
        
        # Log the recording attempt
        app.logger.info(f"Recording attempt by {session.get('username')} for class '{sound_class}' (ambient: {measure_ambient}, unified: {use_unified})")
        
        # Create directories for storing training sounds
        os.makedirs(Config.TRAINING_SOUNDS_DIR, exist_ok=True)
        
        # Create class directory in temp_sounds if it doesn't exist
        temp_class_dir = os.path.join(Config.PENDING_VERIFICATION_SOUNDS_DIR, sound_class)
        os.makedirs(temp_class_dir, exist_ok=True)
        
        # Generate a timestamp for this recording session
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        
        if use_unified:
            # Use the unified audio processing approach
            try:
                from backend.src.services.recording_service import RecordingService
                recording_service = RecordingService()
                
                # Save the audio data to a temporary file
                temp_file = os.path.join(temp_class_dir, f"temp_{timestamp}.wav")
                audio_data.save(temp_file)
                
                # Load the audio with librosa
                import librosa
                audio, sr = librosa.load(temp_file, sr=16000)
                
                # Determine if we should measure ambient noise - default to True for better accuracy
                # Can be overridden by query parameter or based on specific endpoints
                measure_ambient = request.args.get('measure_ambient', 'true').lower() != 'false'
                
                # Process the audio to get segments using the centralized audio preprocessor
                # Use preprocess_recording with ambient noise measurement for better thresholding
                # Pass sample rate and measure_ambient flag to ensure proper preprocessing
                segments = recording_service.preprocessor.preprocess_recording(
                    audio_data=audio,
                    sr=sr,
                    measure_ambient=measure_ambient
                )
                
                # If no segments were detected
                if not segments or len(segments) == 0:
                    # Clean up the temp file
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    
                    return jsonify({
                        'success': False, 
                        'message': 'No sound segments were detected in your recording. Please try again with clearer sounds separated by silence.'
                    })
                
                # Save each segment for verification
                segment_files = []
                for i, segment in enumerate(segments):
                    segment_filename = f"{sound_class}_{timestamp}_{i+1}.wav"
                    segment_path = os.path.join(temp_class_dir, segment_filename)
                    
                    # Save the segment
                    import soundfile as sf
                    sf.write(segment_path, segment, sr)
                    segment_files.append(segment_path)
                
                app.logger.info(f"Saved {len(segment_files)} segments for verification")
                
                # Clean up the temp file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                
                # Redirect to verification page
                return redirect(url_for('unified_sound_verify', class_name=sound_class))
                
            except ImportError as e:
                app.logger.error(f"Failed to import RecordingService: {e}")
                use_unified = False
        
        # Fall back to original method if unified approach not available or fails
        if not use_unified:
            # Save the original recording to a temporary file
            temp_file = os.path.join(temp_class_dir, f"temp_{timestamp}.wav")
            audio_data.save(temp_file)
            
            # Process the audio file to extract sound segments
            from backend.src.ml.sound_detector import process_audio_file
            
            try:
                # Process the audio file and get segments
                segments = process_audio_file(
                    temp_file, 
                    output_dir=temp_class_dir,
                    class_name=sound_class,
                    timestamp=timestamp,
                    check_ambient=measure_ambient
                )
                
                # Clean up the temp file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                
                if not segments or len(segments) == 0:
                    return jsonify({
                        'success': False, 
                        'message': 'No sound segments were detected in your recording. Please try again with clearer sounds separated by silence.'
                    })
                
                # Redirect to verification page
                return redirect(url_for('unified_sound_verify', class_name=sound_class))
                
            except Exception as e:
                app.logger.error(f"Error processing audio: {str(e)}")
                # Clean up the temp file
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                return jsonify({'success': False, 'message': f'Error processing audio: {str(e)}'})
        
    except Exception as e:
        app.logger.error(f"Error in api_ml_record: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})

# --------------------------------------------------------------------
# Consolidated Sound Verification Routes
# --------------------------------------------------------------------
@app.route('/sounds/verify', methods=['GET'])
@app.route('/sounds/verify/<class_name>', methods=['GET'])
def unified_sound_verify(class_name=None):
    """
    Unified sound verification page - handles both single class and all classes
    
    Args:
        class_name: Optional class name to filter (if None, shows all classes)
        
    Returns:
        Rendered template with pending recordings
    """
    try:
        app.logger.info(f"Accessing unified sound verification page with class_name={class_name}")
        
        # Check if user is authenticated
        if 'username' not in session:
            app.logger.warning("User not authenticated, redirecting to login page")
            flash("Please log in to verify sound recordings", "warning")
            return redirect(url_for('login'))
        
        # Variables to track pending recordings
        recordings = []
        pending_count = 0
        
        # Get pending recordings from temp_sounds directory
        pending_dir = Config.PENDING_VERIFICATION_SOUNDS_DIR
        
        app.logger.debug(f"DEBUG: Pending verification directory: {pending_dir}")
        
        if not os.path.exists(pending_dir):
            app.logger.warning(f"Pending verification directory does not exist: {pending_dir}")
        else:
            app.logger.debug(f"Searching for pending recordings in {pending_dir}")
            
            # If a specific class is requested, only look in that folder
            if class_name:
                app.logger.debug(f"Searching for recordings in specific class: {class_name}")
                class_path = os.path.join(pending_dir, class_name)
                
                if os.path.exists(class_path) and os.path.isdir(class_path):
                    # Get all wav files in this class directory
                    for filename in os.listdir(class_path):
                        if filename.endswith('.wav'):
                            # Format file path for the API response
                            file_path = f"{class_name}/{filename}"
                            
                            timestamp = os.path.getmtime(os.path.join(class_path, filename))
                            timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                            
                            recording_data = {
                                'filename': file_path,
                                'id': file_path,  # Use same format as verify.html expects
                                'class_name': class_name,
                                'url': url_for('serve_temp_sound_file', class_name=class_name, filename=filename),
                                'timestamp': timestamp_str
                            }
                            
                            recordings.append(recording_data)
                            app.logger.debug(f"Found pending recording: {file_path}")
                            pending_count += 1
                else:
                    app.logger.warning(f"Class directory not found: {class_path}")
            else:
                # Get all class directories
                class_folders = []
                for item in os.listdir(pending_dir):
                    class_path = os.path.join(pending_dir, item)
                    if os.path.isdir(class_path):
                        class_folders.append(item)
                
                app.logger.debug(f"Found {len(class_folders)} class folders with potential pending recordings: {class_folders}")
                
                # For each class folder, check for pending recordings
                for current_class in class_folders:
                    class_dir = os.path.join(pending_dir, current_class)
                    class_recordings = []
                    
                    app.logger.info(f"Processing class directory: {current_class}")
                    
                    # Count files in this class directory
                    wav_files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
                    app.logger.debug(f"Found {len(wav_files)} WAV files in {current_class} directory")
                    
                    for filename in os.listdir(class_dir):
                        if filename.endswith('.wav'):
                            # Format file path for the API response
                            file_path = f"{current_class}/{filename}"
                            
                            timestamp = os.path.getmtime(os.path.join(class_dir, filename))
                            timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                            
                            recording_data = {
                                'filename': file_path,
                                'id': file_path,  # Use same format as verify.html expects
                                'class_name': current_class,
                                'url': url_for('serve_temp_sound_file', class_name=current_class, filename=filename),
                                'timestamp': timestamp_str
                            }
                            
                            app.logger.debug(f"Adding recording with class_name={recording_data['class_name']}, filename={recording_data['filename']}")
                            class_recordings.append(recording_data)
                            app.logger.debug(f"Found pending recording: {file_path}")
                    
                    if class_recordings:
                        app.logger.debug(f"Adding {len(class_recordings)} recordings for class {current_class}")
                        recordings.extend(class_recordings)
                        pending_count += len(class_recordings)
        
        # Sort recordings by timestamp (newest first)
        recordings.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Debug the recordings list
        app.logger.debug(f"Recordings before grouping: {[rec['class_name'] for rec in recordings]}")
        
        # Group recordings by class for the template
        recordings_by_class = {}
        for recording in recordings:
            current_class = recording['class_name']
            app.logger.debug(f"Grouping recording by class: {current_class}")
            if current_class not in recordings_by_class:
                recordings_by_class[current_class] = []
            recordings_by_class[current_class].append(recording)
        
        app.logger.info(f"Found {pending_count} total pending recordings")
        app.logger.info(f"Classes found: {list(recordings_by_class.keys())}")
        
        # Final sanity check of the data being passed to the template
        for class_key, recordings_list in recordings_by_class.items():
            app.logger.debug(f"Class {class_key} has {len(recordings_list)} recordings")
            for idx, rec in enumerate(recordings_list[:3]):  # Log first 3 recordings per class
                app.logger.debug(f"  - Recording {idx+1}: class_name={rec['class_name']}, filename={rec['filename']}")
        
        # If we're viewing a specific class, filter the recordings
        filtered_recordings = recordings
        if class_name:
            # For class-specific view, we should only pass recordings for that class
            app.logger.debug(f"Filtering recordings for class_name={class_name}")
            filtered_recordings = [r for r in recordings if r['class_name'] == class_name]
            # But still pass all classes in recordings_by_class so the "View All Classes" button works
            app.logger.debug(f"Filtered from {len(recordings)} to {len(filtered_recordings)} recordings")
        
        # Use the unified_verify.html template
        return render_template(
            'unified_verify.html',
            title="Verify Sound Recordings",
            all_classes=(class_name is None),
            class_name=class_name,
            recordings=filtered_recordings,  # Only pass the filtered recordings for the current class
            recordings_by_class=recordings_by_class,  # Pass all classes for dropdown/navigation
        )
    except Exception as e:
        app.logger.error(f"Error in unified_sound_verify route: {str(e)}")
        return f"Error: {str(e)}", 500

# Unified API endpoint for verifying recordings (keep/discard)
@app.route('/api/sounds/verify-recording', methods=['POST'])
def api_verify_recording():
    """
    Unified API endpoint for verifying sound recordings (keep or discard)
    
    Handles both formats:
    - chunkId from verify.html
    - filename from sound_approval.html
    
    Returns:
        JSON response with success/error information
    """
    try:
        data = request.json
        if not data:
            app.logger.error("No data provided in request")
            return jsonify({'success': False, 'error': 'No data provided'})
        
        app.logger.debug(f"Received verification data: {data}")
        
        # Get either filename or chunkId (both formats supported)
        file_id = data.get('filename') or data.get('chunkId')
        if not file_id:
            app.logger.error("No file identifier provided in request")
            return jsonify({'success': False, 'error': 'No file identifier provided'})
            
        # Get keep status
        keep = data.get('keep', False)
        if keep is None and 'isGood' in data:
            keep = data.get('isGood', False)
            
        app.logger.info(f"Processing verification for: {file_id}, keep: {keep}")
        
        # The format should be class_name/filename.wav
        # Make sure we strip any leading/trailing whitespace or extra slashes
        file_id = file_id.strip()
        
        # Check if we have a valid file path
        parts = file_id.split('/', 1)
        if len(parts) != 2:
            app.logger.error(f"Invalid file identifier format: {file_id}")
            return jsonify({'success': False, 'error': f'Invalid file identifier format: {file_id}'})
        
        class_name = parts[0]
        filename_only = parts[1]
        
        app.logger.debug(f"Parsed class_name: {class_name}, filename: {filename_only}")
        
        # Find the source file
        source_path = os.path.join(Config.PENDING_VERIFICATION_SOUNDS_DIR, class_name, filename_only)
        app.logger.debug(f"Looking for file at: {source_path}")
        
        if not os.path.exists(source_path):
            app.logger.error(f"File not found: {source_path}")
            # Try alternate paths
            alt_path = os.path.join(Config.PENDING_VERIFICATION_SOUNDS_DIR, file_id)
            app.logger.debug(f"Trying alternate path: {alt_path}")
            
            if os.path.exists(alt_path):
                app.logger.info(f"Found file at alternate path: {alt_path}")
                source_path = alt_path
            else:
                return jsonify({'success': False, 'error': f'File not found: {source_path}'})
        
        # If keep is true, copy to training directory
        if keep:
            # Create training directory if it doesn't exist
            training_dir = os.path.join(Config.TRAINING_SOUNDS_DIR, class_name)
            os.makedirs(training_dir, exist_ok=True)
            
            # Generate unique filename
            username = session.get('username', 'anonymous')
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            new_filename = f"{class_name}_{username}_{timestamp}_{uuid.uuid4().hex[:6]}.wav"
            
            # Copy to training directory
            training_path = os.path.join(training_dir, new_filename)
            shutil.copy2(source_path, training_path)
            app.logger.info(f"Saved approved recording to {training_path}")
        
        # Delete the source file (regardless of keep/discard)
        try:
            os.remove(source_path)
            app.logger.info(f"Removed source file from pending verification: {source_path}")
        except Exception as del_err:
            app.logger.warning(f"Could not remove source file: {del_err}")
            # Continue even if deletion fails
            
        return jsonify({'success': True, 'message': 'Recording processed successfully'})
    except Exception as e:
        app.logger.error(f"Error in api_verify_recording: {str(e)}")
        traceback.print_exc()  # Print full stack trace to server logs
        return jsonify({'success': False, 'error': str(e)})

# --------------------------------------------------------------------
# Before request: check login
# --------------------------------------------------------------------
@app.before_request
def check_login():
    """
    This is not strictly necessary since we're checking in each route,
    but it's a good precaution for any routes we might forget to protect.
    """
    # Skip login check for static assets, login/register routes, and API routes
    if (request.endpoint in ['login', 'register', 'static', 'logout', 'test_login'] or
        request.path.startswith('/static/') or
        request.path.startswith('/api/') or
        request.path == '/test-login'):
        return None
        
    # Require login for all other routes
    if 'username' not in session:
        if request.path != '/':
            flash('Please login to access this page', 'warning')
        return redirect(url_for('login'))
    
    return None

# --------------------------------------------------------------------
# Error handlers
# --------------------------------------------------------------------
@app.errorhandler(404)
def not_found_error(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    app.logger.error("Server error: %s", e)
    return render_template('500.html', error=str(e)), 500

# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------
# Add a route that will ensure the user is logged out when first accessing the app
@app.route('/auto-logout')
def auto_logout():
    """
    Route to automatically log out the user and redirect to login page.
    This is called when the app first starts to ensure users begin in a logged-out state.
    """
    session.clear()
    flash('You have been logged out for security reasons.', 'info')
    return redirect(url_for('login'))

@app.route('/admin/sync-model-registry')
def sync_model_registry():
    """Admin route to manually trigger synchronization of the models.json registry file"""
    # Check if user is logged in and is an admin
    if 'username' not in session or not session.get('is_admin', False):
        flash('You must be logged in as an admin to access this page.', 'error')
        return redirect(url_for('login'))
    
    try:
        # Call the synchronize_model_registry function
        success = synchronize_model_registry()
        
        if success:
            flash('Model registry synchronized successfully', 'success')
            app.logger.info("Model registry manually synchronized by %s", session.get('username'))
        else:
            flash('Failed to synchronize model registry', 'error')
            app.logger.error("Failed to manually synchronize model registry")
        
        # Redirect to the dashboard or another appropriate page
        return redirect(url_for('index'))
    except Exception as e:
        app.logger.error("Error syncing model registry: %s", str(e))
        flash(f'Error syncing model registry: {str(e)}', 'error')
        return redirect(url_for('index'))

@app.route('/api/sounds/pending/all', methods=['GET'])
def api_all_pending_recordings():
    """
    API endpoint to get all pending recordings across all classes
    """
    try:
        app.logger.info("Received request for all pending recordings")
        
        # Check if user is authenticated
        if 'username' not in session:
            # Temporary debug bypass - automatically set test user
            session['username'] = 'testuser'
            session['user_id'] = 'test_id'
            session['is_admin'] = True
            app.logger.warning("Temporarily set test user in session for API debugging")
            # Commented out authentication check for debugging
            # app.logger.warning("Authentication required for api_all_pending_recordings")
            # return jsonify({
            #     'success': False, 
            #     'message': 'Authentication required'
            # }), 401
        
        # Check pending recordings across all classes
        recordings = []
        pending_count = 0
        
        # Search for all folders in the pending verification directory
        pending_dir = Config.PENDING_VERIFICATION_SOUNDS_DIR
        if os.path.exists(pending_dir):
            app.logger.debug(f"Searching for pending recordings in {pending_dir}")
            
            # Get all class directories
            class_folders = []
            for item in os.listdir(pending_dir):
                class_path = os.path.join(pending_dir, item)
                if os.path.isdir(class_path):
                    class_folders.append(item)
            
            app.logger.debug(f"Found {len(class_folders)} class folders with potential pending recordings")
            
            # For each class folder, check for pending recordings
            for class_name in class_folders:
                class_dir = os.path.join(pending_dir, class_name)
                class_recordings = []
                
                for filename in os.listdir(class_dir):
                    if filename.endswith('.wav'):
                        # Format file path for the API response
                        file_path = f"{class_name}/{filename}"
                        
                        timestamp = os.path.getmtime(os.path.join(class_dir, filename))
                        timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                        
                        recording_data = {
                            'filename': file_path,
                            'class_name': class_name,
                            'url': url_for('serve_temp_sound_file', class_name=class_name, filename=filename),
                            'timestamp': timestamp_str
                        }
                        
                        class_recordings.append(recording_data)
                        app.logger.debug(f"Found pending recording: {file_path}")
                
                if class_recordings:
                    recordings.extend(class_recordings)
                    pending_count += len(class_recordings)
        
        # Sort recordings by timestamp (newest first)
        recordings.sort(key=lambda x: x['timestamp'], reverse=True)
        
        app.logger.info(f"Found {pending_count} total pending recordings across all classes")
        return jsonify({
            'success': True, 
            'recordings': recordings,
            'count': pending_count
        })
    except Exception as e:
        app.logger.error(f"Error in api_all_pending_recordings: {str(e)}")
        return jsonify({
            'success': False, 
            'error': str(e), 
            'recordings': [],
            'count': 0
        })

@app.route('/sounds/approval', methods=['GET'])
def sound_approval():
    """
    Page for approving pending sound recordings across all classes
    Redirects to the new unified verification page
    """
    # Legacy route - redirect to the new unified verification page
    app.logger.info("Legacy route /sounds/approval accessed, redirecting to /sounds/verify")
    return redirect(url_for('unified_sound_verify'))

# Add new route to serve files from temp_sounds
@app.route('/temp_sounds/<class_name>/<filename>')
def serve_temp_sound_file(class_name, filename):
    """
    Route to serve temporary sound files from temp_sounds directory
    """
    try:
        file_path = os.path.join(Config.PENDING_VERIFICATION_SOUNDS_DIR, class_name, filename)
        
        if not os.path.isfile(file_path):
            app.logger.error(f"Temp sound file not found: {file_path}")
            return "File not found", 404
            
        directory = os.path.join(Config.PENDING_VERIFICATION_SOUNDS_DIR, class_name)
        return send_from_directory(directory, filename)
    except Exception as e:
        app.logger.error(f"Error serving temp sound file: {str(e)}")
        return str(e), 500

# Route to serve temporary files
@app.route('/sounds/temp/<path:filename>')
def serve_temp_file(filename):
    """
    Route to serve temporary audio files (used for verification)
    """
    try:
        # Make sure the file actually exists in the temp directory
        full_path = os.path.join(Config.TEMP_DIR, filename)
        if not os.path.isfile(full_path):
            app.logger.error(f"Temp file not found: {full_path}")
            return "File not found", 404
            
        return send_from_directory(Config.TEMP_DIR, filename)
    except Exception as e:
        app.logger.error(f"Error serving temp file: {str(e)}")
        return str(e), 500

# Add a route to serve sound files directly
@app.route('/sounds/<class_name>/<sample_name>')
def serve_sound_file(class_name, sample_name):
    """Serve a sound file."""
    file_path = os.path.join(Config.TRAINING_SOUNDS_DIR, class_name, sample_name)
    
    if not os.path.exists(file_path):
        return "File not found", 404
    
    return send_from_directory(Config.TRAINING_SOUNDS_DIR, os.path.join(class_name, sample_name))

# Legacy route for backward compatibility
@app.route('/verify/<class_name>/<timestamp>')
def verify_sounds(class_name, timestamp):
    """
    Legacy route for verifying sound chunks after recording.
    Redirects to the new unified verification page
    """
    app.logger.info(f"Legacy route /verify/{class_name}/{timestamp} accessed, redirecting to /sounds/verify/{class_name}")
    return redirect(url_for('unified_sound_verify', class_name=class_name))

@app.route('/process_verification', methods=['POST'])
def process_verification():
    """
    Legacy endpoint for processing verification
    Now redirects to the new API endpoint
    """
    app.logger.info("Legacy route /process_verification accessed, redirecting to new API")
    
    # Forward the request to the new API endpoint
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})
            
        # Transform data if needed
        if 'chunkId' in data and 'className' in data:
            # Create properly formatted file_id
            chunk_id = data.get('chunkId')
            class_name = data.get('className')
            
            # If chunkId is already in the format class_name/filename, use it directly
            if '/' in chunk_id:
                file_id = chunk_id
            else:
                # Otherwise construct it from className and chunkId
                file_id = f"{class_name}/{chunk_id}"
                
            # Map isGood to keep
            keep = data.get('isGood', False)
            
            # Forward to the new API
            return api_verify_recording()
        else:
            return jsonify({'success': False, 'error': 'Invalid request format'})
    except Exception as e:
        app.logger.error(f"Error in legacy process_verification: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

# Add a simple test route for debugging the verify page buttons
@app.route('/test-hello-world')
def test_hello_world():
    action = request.args.get('action', 'unknown')
    file = request.args.get('file', 'no-file')
    html = f"""
    <html>
    <head>
        <title>Button Test</title>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; }}
            .container {{ max-width: 800px; margin: 0 auto; background: #f5f5f5; padding: 20px; border-radius: 8px; }}
            h1 {{ color: #333; }}
            .action {{ font-weight: bold; font-size: 1.2em; }}
            .file {{ word-break: break-all; background: #e9e9e9; padding: 10px; border-radius: 4px; }}
            .back-link {{ margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Button Click Test Successful!</h1>
            <p>The button click was detected and routed correctly.</p>
            
            <p>Action: <span class="action">{action}</span></p>
            <p>File: <div class="file">{file}</div></p>
            
            <p>This confirms that the button is working correctly as a link.</p>
            
            <div class="back-link">
                <a href="/sounds/verify">Back to Verify Page</a>
            </div>
        </div>
    </body>
    </html>
    """
    return html

if __name__ == '__main__':
    try:
        print("\n\nStarting script execution - main.py\n\n")
        
        import sys
        sys.stdout.write("Testing immediate output with sys.stdout.write\n")
        sys.stdout.flush()
        
        import argparse
        print("Imported argparse successfully")
        print("Imported sys successfully")
        import traceback
        print("Imported traceback successfully")
        import logging
        print("Imported logging successfully")
    
        try:
            print("Attempting to import Config...")
            from backend.config import Config
            print(f"Successfully imported Config, DATA_DIR={Config.DATA_DIR}")
        except Exception as e:
            print(f"ERROR importing config: {e}")
            traceback.print_exc()
            sys.exit(1)
            
        try:
            print("Attempting to import AudioAugmentor...")
            from backend.audio.augmentor import AudioAugmentor
            print("Successfully imported AudioAugmentor")
        except Exception as e:
            print(f"ERROR importing AudioAugmentor: {e}")
            traceback.print_exc()
            sys.exit(1)
            
        try:
            print("Attempting to import AudioPreprocessor...")
            from backend.audio.preprocessor import AudioPreprocessor
            print("Successfully imported AudioPreprocessor")
        except Exception as e:
            print(f"ERROR importing AudioPreprocessor: {e}")
            traceback.print_exc()
            sys.exit(1)
        
        parser = argparse.ArgumentParser(description='Sound Classifier Application and Utilities')
        parser.add_argument('--generate-augmentation', action='store_true', 
                          help='Generate augmented audio files from original sound files')
        parser.add_argument('--verbose', action='store_true',
                          help='Enable verbose logging')
        
        args = parser.parse_args()
        
        if args.generate_augmentation:
            print("\n--generate-augmentation flag detected")
            sys.stdout.write("Setting up logging...\n")
            sys.stdout.flush()
            
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            root_logger = logging.getLogger()
            root_logger.addHandler(console_handler)
            
            if args.verbose:
                root_logger.setLevel(logging.DEBUG)
                print("Verbose logging enabled")
            else:
                root_logger.setLevel(logging.INFO)
                
            print(f"Logger level: {logging.getLevelName(root_logger.level)}")
            print(f"Logger handlers: {root_logger.handlers}")
            
            sys.stdout.write("Logging setup complete\n")
            sys.stdout.flush()
            
            print("\n" + "=" * 60)
            print("STARTING AUDIO AUGMENTATION PROCESS")
            print("=" * 60)
            
            try:
                print(f"Current directory: {os.getcwd()}")
                print(f"Python version: {sys.version}")
                print(f"Config.DATA_DIR: {Config.DATA_DIR}")
                print(f"Config.LOGS_DIR: {Config.LOGS_DIR}")
                
                if not os.path.exists(Config.DATA_DIR):
                    print(f"WARNING: DATA_DIR doesn't exist: {Config.DATA_DIR}")
                else:
                    print(f"DATA_DIR exists, content count: {len(os.listdir(Config.DATA_DIR))}")
                    
                if not os.path.exists(Config.LOGS_DIR):
                    print(f"WARNING: LOGS_DIR doesn't exist: {Config.LOGS_DIR}")
                    os.makedirs(Config.LOGS_DIR, exist_ok=True)
                    print(f"Created LOGS_DIR: {Config.LOGS_DIR}")
                
                print("Initializing preprocessor...")
                sys.stdout.write("Creating AudioPreprocessor instance...\n")
                sys.stdout.flush()
                preprocessor = AudioPreprocessor(sample_rate=16000)
                print("Preprocessor instance created, methods: " + ", ".join([m for m in dir(preprocessor) if not m.startswith('_')]))
                
                print("Initializing augmentor...")
                sys.stdout.write("Creating AudioAugmentor instance...\n")
                sys.stdout.flush()
                augmentor = AudioAugmentor(sample_rate=16000, preprocessor=preprocessor)
                print("Augmentor instance created, methods: " + ", ".join([m for m in dir(augmentor) if not m.startswith('_')]))
                print("Augmentor initialized successfully")
                
                input_dir = os.path.join(Config.DATA_DIR, 'sounds')
                output_dir = os.path.join(Config.DATA_DIR, 'augmented_sounds')
                
                print(f"Input directory: {input_dir}")
                print(f"Output directory: {output_dir}")
            
                if not os.path.exists(input_dir):
                    print(f"ERROR: Input directory {input_dir} does not exist")
                    sys.exit(1)
                else:
                    sound_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
                    print(f"Found {len(sound_dirs)} sound directories: {', '.join(sound_dirs)}")
                    
                    total_files = 0
                    for sound_dir in sound_dirs:
                        sound_path = os.path.join(input_dir, sound_dir)
                        class_dirs = [d for d in os.listdir(sound_path) if os.path.isdir(os.path.join(sound_path, d))]
                        sound_dir_total = 0
                        for class_dir in class_dirs:
                            class_path = os.path.join(sound_path, class_dir)
                            wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
                            sound_dir_total += len(wav_files)
                            if len(wav_files) > 0:
                                print(f"  - {sound_dir}/{class_dir}: {len(wav_files)} .wav files")
                        print(f"  Total in {sound_dir}: {sound_dir_total} .wav files")
                        total_files += sound_dir_total
                    
                    print(f"Total input files: {total_files}")
                    if total_files == 0:
                        print("ERROR: No input .wav files found in the class directories")
                        sys.exit(1)
            
                print("\nTesting audio preprocessing on a few sample files:")
                sample_files = []
                for sound_dir in sound_dirs[:min(3, len(sound_dirs))]:
                    sound_path = os.path.join(input_dir, sound_dir)
                    class_dirs = [d for d in os.listdir(sound_path) if os.path.isdir(os.path.join(sound_path, d))]
                    for class_dir in class_dirs[:min(3, len(class_dirs))]:
                        class_path = os.path.join(sound_path, class_dir)
                        wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
                        if wav_files:
                            sample_files.append((class_path, wav_files[0]))
                            if len(sample_files) >= 3:
                                break
                    if len(sample_files) >= 3:
                        break
                    
                for class_path, wav_file in sample_files:
                    test_file = os.path.join(class_path, wav_file)
                    print(f"Testing file: {test_file}")
                    try:
                        import librosa
                        audio_data, sr = librosa.load(test_file, sr=16000)
                        print(f"  - Loaded successfully. Length: {len(audio_data)}, Duration: {len(audio_data)/sr:.2f}s")
                        
                        print("  - Testing preprocessor...")
                        processed = preprocessor.preprocess_audio(audio_data)
                        if processed is not None:
                            print(f"  - Preprocessing successful. Length: {len(processed)}")
                        else:
                            print("  - Preprocessing returned None")
                            
                        print("  - Testing augmentation...")
                        augmented = augmentor.augment_audio(audio_data, 'pitch_shift')
                        if augmented is not None:
                            print(f"  - Augmentation successful. Length: {len(augmented)}")
                        else:
                            print("  - Augmentation returned None")
                    except Exception as e:
                        print(f"  - Error testing file: {str(e)}")
                    
                os.makedirs(output_dir, exist_ok=True)
                print(f"\nOutput directory created: {output_dir}")
                
                print("\nStarting directory augmentation...")
                sys.stdout.write("Calling augmentor.augment_directory...\n")
                sys.stdout.flush()
                print(f"augment_directory parameters: input_dir={input_dir}, output_dir={output_dir}, count_per_file=5, class_specific=True")
                
                try:
                    stats = augmentor.augment_directory(
                        input_dir=input_dir,
                        output_dir=output_dir,
                        count_per_file=5,  # Generate 5 augmentations per original file
                        class_specific=True  # Maintain class folder structure
                    )
                    print("augment_directory completed successfully")
                except Exception as e:
                    print(f"ERROR in augment_directory: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    raise  # Re-raise for outer error handling
                
                print("\n" + "=" * 60)
                print("AUGMENTATION COMPLETE")
                print("=" * 60)
                print(f"Processed {stats['input_files']} input files")
                print(f"Generated {stats['augmented_files']} augmented files")
                print(f"Failed files: {stats['failed_files']}")
                
                if stats['augmented_files'] > 0:
                    success_rate = (stats['input_files'] - stats['failed_files']) / stats['input_files'] * 100
                    print(f"Overall success rate: {success_rate:.1f}%")
                    print(f"Augmented files saved to: {output_dir}")
                else:
                    print("ERROR: No augmented files were generated.")
                    for class_name, class_stats in stats['classes'].items():
                        print(f"Class '{class_name}': {class_stats['augmented_files']} files created from {class_stats['input_files']} input files, {class_stats['failed_files']} failures")
                    sys.exit(1)
                
                sys.exit(0)
            
            except Exception as e:
                print(f"\nERROR during augmentation process: {str(e)}")
                print(f"Error type: {type(e).__name__}")
                print(f"Error args: {e.args}")
                traceback.print_exc()
                
                if "No module named" in str(e):
                    print(f"\nThis appears to be a missing module error. Please install the required module.")
                    print(f"You can try: pip install {str(e).split('No module named')[1].strip()}")
                elif "Permission denied" in str(e):
                    print(f"\nThis appears to be a file permission error. Check file permissions for the relevant directories.")
                
                sys.exit(1)
        
        else:
            try:
                print("Starting Flask app...")
                # The app is already defined in this file, no need to import it
                app.run(debug=Config.DEBUG, host='0.0.0.0', port=5002)
            except Exception as e:
                print(f"ERROR starting Flask app: {e}")
                traceback.print_exc()
                sys.exit(1)
    
    except Exception as e:
        print(f"\nCRITICAL: Unhandled exception at top level: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
