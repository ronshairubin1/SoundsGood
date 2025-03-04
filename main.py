import os
import json
import uuid
import shutil
import logging
import inspect
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from flask_cors import CORS
from werkzeug.utils import secure_filename
import time
from config import Config

# Configure logging
logging.basicConfig(
    filename=os.path.join(Config.LOGS_DIR, 'app_execution.log'),
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('main')
logger.info("=" * 50)
logger.info("Starting Sound Classifier app with new architecture...")

logger.info(f"Loaded Config from config.py")

# Import our new API instead of the old routes
try:
    from src.api.ml_api import MlApi
    logger.info("Imported MlApi from src.api.ml_api")
except ImportError as e:
    logger.error(f"Failed to import MlApi: {e}")
    import sys
    logger.critical("MlApi is a critical component. Exiting application due to import failure.")
    sys.exit(1)

try:
    from src.api.dictionary_api import dictionary_bp
    logger.info("Imported dictionary_bp from src.api.dictionary_api")
except ImportError as e:
    logger.error(f"Failed to import dictionary_bp: {e}")

try:
    from src.api.user_api import user_bp
    logger.info("Imported user_bp from src.api.user_api")
except ImportError as e:
    logger.error(f"Failed to import user_bp: {e}")

try:
    from src.api.dashboard_api import dashboard_bp
    logger.info("Imported dashboard_bp from src.api.dashboard_api")
except ImportError as e:
    logger.error(f"Failed to import dashboard_bp: {e}")

# Import the ml_bp from routes
try:
    from src.routes.ml_routes import ml_bp
    logger.info("Imported ml_bp from src.routes.ml_routes")
except ImportError as e:
    logger.error(f"Failed to import ml_bp: {e}")

# Import services
try:
    from src.services.dictionary_service import DictionaryService
    logger.info("Imported DictionaryService from src.services.dictionary_service")
except ImportError as e:
    logger.error(f"Failed to import DictionaryService: {e}")

try:
    from src.services.user_service import UserService
    logger.info("Imported UserService from src.services.user_service")
except ImportError as e:
    logger.error(f"Failed to import UserService: {e}")

# --------------------------------------------------------------------
# Set up Flask
# --------------------------------------------------------------------
app = Flask(
    __name__,
    static_url_path='/static',
    static_folder=os.path.join(Config.BASE_DIR, 'static'),
    template_folder=os.path.join(Config.BASE_DIR, 'src', 'templates')
)
logger.debug(f"Template folder: {os.path.join(Config.BASE_DIR, 'src', 'templates')}")
logger.debug(f"Static folder: {os.path.join(Config.BASE_DIR, 'static')}")

app.secret_key = Config.SECRET_KEY
CORS(app, supports_credentials=True)

# Ensure required directories exist
os.makedirs(Config.TEMP_DIR, exist_ok=True)
logger.debug(f"Ensured TEMP_DIR exists: {Config.TEMP_DIR}")
os.makedirs(Config.TRAINING_SOUNDS_DIR, exist_ok=True)
logger.debug(f"Ensured TRAINING_SOUNDS_DIR exists: {Config.TRAINING_SOUNDS_DIR}")

# Register blueprints
try:
    app.register_blueprint(dictionary_bp)
    logger.info("Registered dictionary_bp blueprint")
    app.register_blueprint(user_bp)
    logger.info("Registered user_bp blueprint")
    app.register_blueprint(dashboard_bp) 
    logger.info("Registered dashboard_bp blueprint")
    app.register_blueprint(ml_bp)  # Register the ml_bp
    logger.info("Registered ml_bp blueprint")
except Exception as e:
    logger.error(f"Error registering blueprints: {e}")

# Initialize services
dictionary_service = DictionaryService()
logger.info("Initialized DictionaryService")
user_service = UserService()
logger.info("Initialized UserService")

# Initialize the ML API with our Flask app
ml_api = MlApi(app, model_dir=Config.MODELS_DIR)

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
app.logger.debug(f"Template folder: {app.template_folder}")
app.logger.debug(f"Static folder: {app.static_folder}")

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
        app.logger.info(f"Created sounds directory: {Config.TRAINING_SOUNDS_DIR}")
    
    # Create the temp directory if it doesn't exist
    if not os.path.exists(Config.TEMP_DIR):
        os.makedirs(Config.TEMP_DIR, exist_ok=True)
        app.logger.info(f"Created temp directory: {Config.TEMP_DIR}")

# Initialize application directories
init_app_directories()

# --------------------------------------------------------------------
# Basic Routes: index, login, logout, register
# --------------------------------------------------------------------
@app.route('/')
def index():
    # If not logged in, ask for login
    if 'username' not in session:
        return render_template('login.html')
    
    # If logged in, go to dashboard with stats
    # Prepare dashboard stats
    stats = {
        'dictionaries': 0,
        'classes': 0,
        'recordings': 0,
        'models': 0
    }
    
    # Get statistics from services if possible
    try:
        # Count dictionaries
        dictionaries = dictionary_service.get_dictionaries(session.get('user_id'))
        stats['dictionaries'] = len(dictionaries) if dictionaries else 0
        
        # Count classes across all dictionaries
        all_classes = set()  # Use a set to avoid duplicates
        
        for dict_info in dictionaries or []:
            # Check for classes field first, then sounds field as fallback
            class_list = []
            if 'classes' in dict_info and dict_info['classes']:
                class_list = dict_info['classes']
            elif 'sounds' in dict_info and dict_info['sounds']:
                class_list = dict_info['sounds']
            
            # Add all classes to our set
            all_classes.update(class_list)
        
        stats['classes'] = len(all_classes)
        
        # Count actual wav files in the sounds directory
        total_recordings = 0
        for class_name in all_classes:
            class_path = os.path.join(Config.TRAINING_SOUNDS_DIR, class_name)
            if os.path.exists(class_path) and os.path.isdir(class_path):
                recordings = [f for f in os.listdir(class_path) if f.endswith('.wav') or f.endswith('.mp3')]
                total_recordings += len(recordings)
                
        stats['recordings'] = total_recordings
        
        app.logger.info(f"Dashboard stats: {stats}")
        
        # Count models (by checking files in models directory)
        model_files = [f for f in os.listdir(Config.MODELS_DIR) 
                     if f.endswith('.h5') or f.endswith('.joblib')]
        stats['models'] = len(model_files)
    except Exception as e:
        app.logger.error(f"Error getting dashboard stats: {e}")
    
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
            flash(f'Welcome back, {username}!', 'success')
            return redirect(url_for('index'))
        else:
            flash(result['error'], 'danger')
            return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

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
        else:
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
    app.logger.info(f"Fetching dictionaries for user: {session.get('user_id')}")
    
    # Explicitly reload the dictionary service
    app.logger.info("Force reloading dictionary service data")
    dictionary_service._load_dictionaries()
    
    dictionaries = dictionary_service.get_dictionaries(session.get('user_id'))
    app.logger.info(f"Found {len(dictionaries)} dictionaries: {[d.get('name') for d in dictionaries]}")
    
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
        app.logger.debug(f"Dictionary classes: {dictionary['classes']}")
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
    response.headers.set('Content-Disposition', f'attachment; filename={dict_name.replace(" ", "_").lower()}_export.json')
    return response

# --------------------------------------------------------------------
# Training and Prediction Routes
# --------------------------------------------------------------------
@app.route('/training')
def training():
    """Render the training page."""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get dictionaries for selecting which to train on
    dictionaries = dictionary_service.get_dictionaries(session.get('user_id'))
    
    # Get selected dictionary if provided in query string
    selected_dict = request.args.get('dictionary')
    
    return render_template('training.html', 
                          dictionaries=dictionaries,
                          selected_dict=selected_dict)

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
    
    # Get all sound classes from the sounds directory
    training_sounds_dir = Config.TRAINING_SOUNDS_DIR
    sound_classes = []
    
    if os.path.exists(training_sounds_dir):
        # Get directories representing sound classes
        class_dirs = [d for d in os.listdir(training_sounds_dir) 
                     if os.path.isdir(os.path.join(training_sounds_dir, d))]
        
        # Get sample counts for each class
        for class_name in class_dirs:
            class_path = os.path.join(training_sounds_dir, class_name)
            samples = [f for f in os.listdir(class_path) if f.lower().endswith('.wav')]
            
            sound_classes.append({
                'name': class_name,
                'sample_count': len(samples),
                'path': class_path
            })
    
    return render_template('sounds_management.html', sound_classes=sound_classes)

@app.route('/sounds/record')
def record_sounds():
    """Render the sounds recording page."""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get all sound classes from the sounds directory for the dropdown
    training_sounds_dir = Config.TRAINING_SOUNDS_DIR
    sound_classes = []
    
    if os.path.exists(training_sounds_dir):
        # Get directories representing sound classes
        class_dirs = [d for d in os.listdir(training_sounds_dir) 
                     if os.path.isdir(os.path.join(training_sounds_dir, d))]
        
        # Get sample counts for each class
        for class_name in class_dirs:
            class_path = os.path.join(training_sounds_dir, class_name)
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
                          dict_classes=dict_classes)

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
    """API endpoint for getting sound classes."""
    training_sounds_dir = Config.TRAINING_SOUNDS_DIR
    
    result = []
    if os.path.exists(training_sounds_dir):
        app.logger.debug(f"Reading sound classes from {training_sounds_dir}")
        class_dirs = [d for d in os.listdir(training_sounds_dir)
                      if os.path.isdir(os.path.join(training_sounds_dir, d))]
        
        for class_name in sorted(class_dirs):
            class_path = os.path.join(training_sounds_dir, class_name)
            samples = [f for f in os.listdir(class_path) if f.lower().endswith('.wav')]
            
            result.append({
                'name': class_name,
                'sample_count': len(samples)
            })
    
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
            'error': f'Class "{class_name}" already exists'
        }), 400
    
    try:
        # Create directory in main sounds folder
        os.makedirs(class_path, exist_ok=True)
        app.logger.info(f"Created class directory in TRAINING_SOUNDS_DIR: {class_path}")
        
        # Create directories in data/sounds subdirectories
        # 1. Raw sounds directory
        raw_class_path = os.path.join(Config.RAW_SOUNDS_DIR, safe_class_name)
        os.makedirs(raw_class_path, exist_ok=True)
        app.logger.info(f"Created class directory in RAW_SOUNDS_DIR: {raw_class_path}")
        
        # 2. Temp sounds directory
        temp_class_path = os.path.join(Config.PENDING_VERIFICATION_SOUNDS_DIR, safe_class_name)
        os.makedirs(temp_class_path, exist_ok=True)
        app.logger.info(f"Created class directory in pending verification sounds dir: {temp_class_path}")
        
        # 3. Training sounds directory
        training_class_path = os.path.join(Config.TRAINING_SOUNDS_DIR, safe_class_name)
        os.makedirs(training_class_path, exist_ok=True)
        app.logger.info(f"Created class directory in TRAINING_SOUNDS_DIR: {training_class_path}")
        
        # Also update the classes.json file if it exists
        classes_dir = os.path.join(os.path.dirname(__file__), 'data', 'classes')
        classes_json_path = os.path.join(classes_dir, 'classes.json')
        
        if os.path.exists(os.path.dirname(classes_json_path)):
            # Ensure the classes directory exists
            os.makedirs(os.path.dirname(classes_json_path), exist_ok=True)
            
            classes_data = {"classes": {}}
            if os.path.exists(classes_json_path):
                try:
                    with open(classes_json_path, 'r') as f:
                        classes_data = json.load(f)
                except Exception as e:
                    app.logger.error(f"Error loading classes.json: {e}")
            
            # Add the new class entry
            classes_data['classes'][safe_class_name] = {
                "name": safe_class_name,
                "samples": [],
                "sample_count": 0,
                "created_at": datetime.now().isoformat(),
                "in_dictionaries": []
            }
            
            # Save the updated classes.json
            try:
                with open(classes_json_path, 'w') as f:
                    json.dump(classes_data, f, indent=4)
                app.logger.info(f"Updated classes.json with new class: {safe_class_name}")
            except Exception as e:
                app.logger.error(f"Error updating classes.json: {e}")
        
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
            'error': f'Class "{class_name}" not found'
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
            app.logger.error(f"Error updating dictionaries after class deletion: {dict_error}")
        
        return jsonify({
            'success': True,
            'message': f'Class "{class_name}" and all its recordings deleted successfully'
        })
    except Exception as e:
        app.logger.error(f"Error deleting sound class: {e}")
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
            'error': f'Class "{class_name}" not found'
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
    API endpoint to process a sound recording through the ML pipeline
    """
    try:
        # Check if user is authenticated
        if 'username' not in session:
            return jsonify({
                'success': False, 
                'message': 'Authentication required'
            }), 401
        
        # Get form data
        audio_data = request.files.get('audio')
        sound_class = request.form.get('sound')
        measure_ambient = request.form.get('measure_ambient', 'false').lower() == 'true'
        
        # Log the recording attempt
        app.logger.info(f"Recording attempt: user={session['username']}, class={sound_class}, measure_ambient={measure_ambient}")
        
        # Check for required data
        if not sound_class:
            return jsonify({
                'success': False, 
                'message': 'No sound class specified'
            }), 400
        
        if not audio_data:
            return jsonify({
                'success': False, 
                'message': 'No audio data received'
            }), 400
        
        # Create timestamp for this recording
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        
        # Create directories for raw and temp sounds
        raw_sounds_dir = os.path.join(Config.RAW_SOUNDS_DIR, sound_class)
        temp_sounds_dir = os.path.join(Config.PENDING_VERIFICATION_SOUNDS_DIR, sound_class)
        training_sounds_dir = os.path.join(Config.TRAINING_SOUNDS_DIR, sound_class)
        
        # Ensure the class directories exist - this creates the class if needed
        os.makedirs(raw_sounds_dir, exist_ok=True)
        os.makedirs(temp_sounds_dir, exist_ok=True)
        os.makedirs(training_sounds_dir, exist_ok=True)
        
        # Check if we need to update the classes.json file
        created_new_class = False
        if not os.path.exists(training_sounds_dir) or os.path.getmtime(training_sounds_dir) > time.time() - 2:  # If directory was just created
            created_new_class = True
            
            # Update classes.json if the class was just created
            classes_dir = os.path.join(os.path.dirname(__file__), 'data', 'classes')
            classes_json_path = os.path.join(classes_dir, 'classes.json')
            
            if os.path.exists(os.path.dirname(classes_json_path)):
                # Ensure the classes directory exists
                os.makedirs(os.path.dirname(classes_json_path), exist_ok=True)
                
                classes_data = {"classes": {}}
                if os.path.exists(classes_json_path):
                    try:
                        with open(classes_json_path, 'r') as f:
                            classes_data = json.load(f)
                    except Exception as e:
                        app.logger.error(f"Error loading classes.json: {e}")
                
                # Check if class already exists in the JSON
                if sound_class not in classes_data['classes']:
                    # Add the new class entry
                    classes_data['classes'][sound_class] = {
                        "name": sound_class,
                        "samples": [],
                        "sample_count": 0,
                        "created_at": datetime.now().isoformat(),
                        "in_dictionaries": []
                    }
                    
                    # Save the updated classes.json
                    try:
                        with open(classes_json_path, 'w') as f:
                            json.dump(classes_data, f, indent=4)
                        app.logger.info(f"Updated classes.json with new class: {sound_class}")
                    except Exception as e:
                        app.logger.error(f"Error updating classes.json: {e}")
        
        # Save the original recording to raw_sounds directory
        raw_filename = f"{sound_class}_raw_{timestamp}.webm"
        raw_path = os.path.join(raw_sounds_dir, raw_filename)
        audio_data.save(raw_path)
        
        # Convert audio to WAV format
        wav_filename = f"{sound_class}_raw_{timestamp}.wav"
        wav_path = os.path.join(raw_sounds_dir, wav_filename)
        
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(raw_path)
            audio.export(wav_path, format="wav")
            app.logger.info(f"Converted audio to WAV: {wav_path}")
        except Exception as e:
            app.logger.error(f"Error converting audio: {str(e)}")
            return jsonify({
                'success': False, 
                'message': f'Error converting audio: {str(e)}'
            }), 500
        
        # Chop the recording into chunks using the AudioProcessor
        try:
            from src.core.audio.processor import AudioProcessor
            
            # Initialize with parameters from Config
            processor = AudioProcessor(
                sample_rate=Config.SAMPLE_RATE,
                sound_threshold=Config.SOUND_THRESHOLD,
                silence_threshold=Config.SILENCE_THRESHOLD,
                min_chunk_duration=Config.MIN_CHUNK_DURATION,
                min_silence_duration=Config.MIN_SILENCE_DURATION,
                max_silence_duration=Config.MAX_SILENCE_DURATION,
                sound_multiplier=Config.SOUND_MULTIPLIER,
                sound_end_multiplier=Config.SOUND_END_MULTIPLIER,
                padding_duration=Config.PADDING_DURATION,
                enable_stretching=Config.ENABLE_STRETCHING,
                target_chunk_duration=Config.TARGET_CHUNK_DURATION,
                auto_stop_after_silence=Config.AUTO_STOP_AFTER_SILENCE
            )
            
            # Chop the recording into chunks - output directly to temp_sounds directory
            chunk_files = processor.chop_recording(
                file_path=wav_path,
                output_dir=temp_sounds_dir,
                min_samples=4000,
                use_ambient_noise=measure_ambient
            )
            
            # Log number of chunks created
            app.logger.info(f"Created {len(chunk_files)} chunk(s) from recording")
            
            # Ensure all chunks have the class name prefix
            renamed_chunks = []
            for chunk_file in chunk_files:
                # Get just the basename of the chunk file
                chunk_basename = os.path.basename(chunk_file)
                # Make sure we prefix with the class name 
                if not chunk_basename.startswith(sound_class):
                    new_name = f"{sound_class}_{chunk_basename}"
                    new_path = os.path.join(temp_sounds_dir, new_name)
                    os.rename(chunk_file, new_path)
                    app.logger.info(f"Renamed chunk from {chunk_file} to {new_path}")
                    renamed_chunks.append(new_path)
                else:
                    renamed_chunks.append(chunk_file)
            
            # Check if any chunks were created
            if not renamed_chunks:
                flash('No sound samples were detected in your recording', 'warning')
                return redirect(url_for('record_sounds'))
            
            # Store timestamp in session for verification
            session['verification_timestamp'] = timestamp
            session['verification_class'] = sound_class
            
            # Redirect to the verification page with class and timestamp
            return redirect(url_for('verify_chunks', timestamp=timestamp, class_name=sound_class))
            
        except Exception as e:
            app.logger.error(f"Error chopping recording: {str(e)}")
            return jsonify({
                'success': False, 
                'message': f'Error processing recording: {str(e)}'
            }), 500
    except Exception as e:
        app.logger.error(f"Error in api_ml_record: {str(e)}")
        return jsonify({
            'success': False, 
            'message': f'Error: {str(e)}'
        }), 500

# Routes for verifying and processing sound chunks
@app.route('/verify/<timestamp>')
def verify_chunks(timestamp):
    """
    Route to verify chunks from a recording
    """
    # Check if user is logged in
    if 'username' not in session:
        flash('Please log in to verify recordings', 'warning')
        return redirect(url_for('login'))
    
    # Get class name from query parameters or session
    class_name = request.args.get('class_name') or session.get('verification_class')
    
    if not class_name:
        flash('No class name provided for verification', 'warning')
        return redirect(url_for('record_sounds'))
    
    # Check for chunks in the temp_sounds directory for this class
    temp_sounds_dir = os.path.join(Config.PENDING_VERIFICATION_SOUNDS_DIR, class_name)
    
    if not os.path.isdir(temp_sounds_dir):
        flash('No chunks found for verification', 'warning')
        return redirect(url_for('record_sounds'))
    
    # Get all WAV files in the directory that match the timestamp
    chunks = []
    
    # Find all chunks that match this timestamp in temp_sounds directory
    for filename in os.listdir(temp_sounds_dir):
        if filename.endswith('.wav') and timestamp in filename:
            chunk_id = f"{class_name}/{filename}"
            chunks.append({
                'id': chunk_id,
                'url': url_for('serve_temp_sound_file', class_name=class_name, filename=filename)
            })
            app.logger.info(f"Found chunk for verification: {filename}")
    
    if not chunks:
        flash('No sound samples were detected in your recording', 'warning')
        return redirect(url_for('record_sounds'))
    
    app.logger.info(f"Showing {len(chunks)} chunks for verification of class {class_name}")
    
    # Render the verification template with the chunks
    return render_template('verify.html', chunks=chunks, class_name=class_name)
    
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

@app.route('/process_verification', methods=['POST'])
def process_verification():
    """
    Process the verification of a sound chunk.
    
    This endpoint is called when the user decides whether to keep or discard
    a sound chunk. If kept, the chunk is moved to the appropriate class directory.
    If discarded, the chunk is simply removed.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'})
        
        chunk_id = data.get('chunkId')
        class_name = data.get('className')
        is_good = data.get('isGood', False)
        
        if not chunk_id or not class_name:
            return jsonify({'success': False, 'error': 'Missing required parameters'})
        
        # Find the chunk file
        chunk_path = os.path.join(Config.PENDING_VERIFICATION_SOUNDS_DIR, class_name, chunk_id)
        
        if not os.path.exists(chunk_path):
            return jsonify({'success': False, 'error': f'Chunk file not found: {chunk_path}'})
        
        if is_good:
            # User wants to keep this sample - move it to the appropriate class directory
            # Create the training sounds directory if it doesn't exist
            training_dir = os.path.join(Config.TRAINING_SOUNDS_DIR, class_name)
            os.makedirs(training_dir, exist_ok=True)
            
            # Generate a unique filename
            username = session.get('username', 'anonymous')
            timestamp_now = datetime.now().strftime('%Y%m%d%H%M%S')
            new_filename = f"{class_name}_{username}_{timestamp_now}_{uuid.uuid4().hex[:6]}.wav"
            
            # Save to training sounds directory only (no need for redundant checks)
            training_path = os.path.join(training_dir, new_filename)
            shutil.copy2(chunk_path, training_path)
            app.logger.info(f"Saved approved chunk to {training_path}")
        
        # Always remove the chunk from the temp directory after processing
        try:
            os.remove(chunk_path)
            app.logger.info(f"Removed chunk file: {chunk_path}")
        except Exception as e:
            app.logger.error(f"Error removing chunk file: {str(e)}")
        
        return jsonify({'success': True})
    except Exception as e:
        app.logger.error(f"Error processing verification: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

# Route to serve temporary files
@app.route('/temp/<path:filename>')
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
    if (request.endpoint in ['login', 'register', 'static', 'logout'] or
        request.path.startswith('/static/') or
        request.path.startswith('/api/')):
        return
        
    # Require login for all other routes
    if 'username' not in session:
        if request.path != '/':
            flash('Please login to access this page', 'warning')
        return redirect(url_for('login'))

# --------------------------------------------------------------------
# Error handlers
# --------------------------------------------------------------------
@app.errorhandler(404)
def not_found_error(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    app.logger.error(f"Server error: {e}")
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

if __name__ == '__main__':
    app.run(debug=Config.DEBUG, host='0.0.0.0', port=5002)