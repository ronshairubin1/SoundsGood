import os
import json
import uuid
import shutil
import logging
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from flask_cors import CORS

from config import Config

# Import our new API instead of the old routes
from src.api.ml_api import MlApi
from src.api.dictionary_api import dictionary_bp
from src.api.user_api import user_bp
from src.api.dashboard_api import dashboard_bp

# Import services
from src.services.dictionary_service import DictionaryService
from src.services.user_service import UserService

# --------------------------------------------------------------------
# Set up Flask
# --------------------------------------------------------------------
app = Flask(
    __name__,
    static_url_path='/static',
    static_folder=os.path.join(Config.BASE_DIR, 'static'),
    template_folder=os.path.join(Config.BASE_DIR, 'src', 'templates')
)
app.secret_key = Config.SECRET_KEY
CORS(app, supports_credentials=True)

# Ensure required directories exist
os.makedirs(Config.TEMP_DIR, exist_ok=True)
os.makedirs(Config.SOUNDS_DIR, exist_ok=True)

# Register blueprints - NOTE: we're not using the old ml_bp anymore
app.register_blueprint(dictionary_bp)
app.register_blueprint(user_bp)
app.register_blueprint(dashboard_bp)

# Initialize services
dictionary_service = DictionaryService()
user_service = UserService()

# Initialize the ML API with our Flask app
ml_api = MlApi(app, model_dir=Config.MODELS_DIR)

# --------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------
logging.basicConfig(level=logging.DEBUG, format=Config.LOG_FORMAT)
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
    if not os.path.exists(Config.SOUNDS_DIR):
        os.makedirs(Config.SOUNDS_DIR, exist_ok=True)
        app.logger.info(f"Created sounds directory: {Config.SOUNDS_DIR}")
    
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
        all_classes = []
        total_recordings = 0
        for dict_info in dictionaries or []:
            if 'classes' in dict_info:
                all_classes.extend(dict_info['classes'])
            if 'sample_count' in dict_info:
                total_recordings += dict_info['sample_count']
        
        stats['classes'] = len(all_classes)
        stats['recordings'] = total_recordings
        
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
@app.route('/dictionaries/manage')
def manage_dictionaries():
    """Render the dictionary management page."""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get dictionaries
    dictionaries = dictionary_service.get_dictionaries(session.get('user_id'))
    
    return render_template('manage_dictionaries.html', dictionaries=dictionaries)

@app.route('/dictionaries/<dict_name>/view')
def view_dictionary(dict_name):
    """Render the dictionary view page."""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get dictionary - force reload metadata first
    app.logger.debug(f"Viewing dictionary: {dict_name}")
    
    # Force reload of metadata to ensure we have the latest data
    dictionary_service._load_metadata()
    
    # Get the dictionary with fresh data
    dictionary = dictionary_service.get_dictionary(dict_name)
    app.logger.debug(f"Dictionary data: {dictionary}")
    
    if not dictionary:
        flash(f'Dictionary "{dict_name}" not found', 'danger')
        return redirect(url_for('manage_dictionaries'))
    
    # Log classes for debugging
    if dictionary.get('classes'):
        app.logger.debug(f"Dictionary classes: {dictionary['classes']}")
    else:
        app.logger.debug("Dictionary has no classes defined")
    
    return render_template('dictionary_view.html', dictionary=dictionary)

@app.route('/dictionaries/<dict_name>/edit')
def edit_dictionary(dict_name):
    """Render the dictionary edit page."""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Force reload of metadata to ensure we have the latest data
    dictionary_service._load_metadata()
    
    # Get dictionary
    dictionary = dictionary_service.get_dictionary(dict_name)
    
    if not dictionary:
        flash(f'Dictionary "{dict_name}" not found', 'danger')
        return redirect(url_for('manage_dictionaries'))
    
    return render_template('dictionary_edit.html', dictionary=dictionary)

@app.route('/dictionaries/<dict_name>/export')
def export_dictionary(dict_name):
    """Export a dictionary as JSON."""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get dictionary
    dictionary = dictionary_service.get_dictionary(dict_name)
    
    if not dictionary:
        flash(f'Dictionary "{dict_name}" not found', 'danger')
        return redirect(url_for('manage_dictionaries'))
    
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
    sounds_dir = Config.SOUNDS_DIR
    sound_classes = []
    
    if os.path.exists(sounds_dir):
        # Get directories representing sound classes
        class_dirs = [d for d in os.listdir(sounds_dir) 
                     if os.path.isdir(os.path.join(sounds_dir, d))]
        
        # Get sample counts for each class
        for class_name in class_dirs:
            class_path = os.path.join(sounds_dir, class_name)
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
    sounds_dir = Config.SOUNDS_DIR
    sound_classes = []
    
    if os.path.exists(sounds_dir):
        # Get directories representing sound classes
        class_dirs = [d for d in os.listdir(sounds_dir) 
                     if os.path.isdir(os.path.join(sounds_dir, d))]
        
        # Get sample counts for each class
        for class_name in class_dirs:
            class_path = os.path.join(sounds_dir, class_name)
            samples = [f for f in os.listdir(class_path) if f.lower().endswith('.wav')]
            
            sound_classes.append({
                'name': class_name,
                'sample_count': len(samples),
                'path': class_path
            })
        
        # Sort sound classes alphabetically by name
        sound_classes.sort(key=lambda x: x['name'].lower())
    
    return render_template('sounds_record.html', sound_classes=sound_classes)

@app.route('/sounds/class/<class_name>')
def view_sound_class(class_name):
    """Render the sound class view page."""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get samples for the class
    class_path = os.path.join(Config.SOUNDS_DIR, class_name)
    
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
    """Get all sound classes."""
    sounds_dir = Config.SOUNDS_DIR
    sound_classes = []
    
    if os.path.exists(sounds_dir):
        class_dirs = [d for d in os.listdir(sounds_dir) 
                     if os.path.isdir(os.path.join(sounds_dir, d))]
        
        for class_name in class_dirs:
            class_path = os.path.join(sounds_dir, class_name)
            samples = [f for f in os.listdir(class_path) if f.lower().endswith('.wav')]
            
            sound_classes.append({
                'name': class_name,
                'sample_count': len(samples)
            })
    
    return jsonify({
        'success': True,
        'classes': sound_classes
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
    
    # Create class directory in sounds folder
    class_path = os.path.join(Config.SOUNDS_DIR, safe_class_name)
    
    if os.path.exists(class_path):
        return jsonify({
            'success': False,
            'error': f'Class "{class_name}" already exists'
        }), 400
    
    try:
        os.makedirs(class_path, exist_ok=True)
        
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
    class_path = os.path.join(Config.SOUNDS_DIR, class_name)
    
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
    class_path = os.path.join(Config.SOUNDS_DIR, class_name)
    
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
            
        sample_path = os.path.join(Config.SOUNDS_DIR, class_name, sample_name)
        
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
            
        # Check if there are any pending recordings in the temp directory
        recordings = []
        
        # Loop through directories in the temp folder (each directory is a timestamp)
        if os.path.exists(Config.TEMP_DIR):
            app.logger.debug(f"Searching for pending recordings in {Config.TEMP_DIR}")
            for timestamp_dir in os.listdir(Config.TEMP_DIR):
                timestamp_path = os.path.join(Config.TEMP_DIR, timestamp_dir)
                if os.path.isdir(timestamp_path):
                    # Check each file in the timestamp directory
                    for filename in os.listdir(timestamp_path):
                        # Two ways files could be named - either starting with class_name or
                        # containing the class_name in the filename
                        if filename.endswith('.wav') and (
                            filename.startswith(f"{class_name}_") or 
                            f"_{class_name}_" in filename or
                            f"chunk_{class_name}" in filename):
                            
                            file_path = os.path.join(timestamp_dir, filename)
                            full_path = os.path.join(Config.TEMP_DIR, file_path)
                            
                            # Only include if the file actually exists
                            if os.path.exists(full_path):
                                app.logger.debug(f"Found pending recording: {file_path}")
                                recordings.append({
                                    'filename': file_path,
                                    'class_name': class_name,
                                    'timestamp': timestamp_dir,
                                    'url': url_for('serve_temp_file', filename=file_path)
                                })
        
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
        
        # If the keep status is true, move the file to the class folder
        # Otherwise, delete the file
        timestamp_dir = os.path.dirname(filename) if '/' in filename else 'temp'
        filename_only = os.path.basename(filename)
        
        # Determine the source file path - check if it contains the full path
        if '/' in filename:
            source_path = os.path.join(Config.TEMP_DIR, filename)
        else:
            source_path = os.path.join(Config.TEMP_DIR, timestamp_dir, filename_only)
        
        if not os.path.exists(source_path):
            return jsonify({'success': False, 'error': f'File not found: {source_path}'})
        
        # Extract the class name from the filename
        # Format is typically: classname_timestamp_index.wav
        parts = filename_only.split('_')
        if len(parts) < 2:
            return jsonify({'success': False, 'error': 'Invalid filename format'})
        
        class_name = parts[0]
        
        if keep:
            # Create the class directory if it doesn't exist
            class_dir = os.path.join(Config.SOUNDS_DIR, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Generate a unique filename
            target_filename = f"{class_name}_{uuid.uuid4().hex[:8]}.wav"
            target_path = os.path.join(class_dir, target_filename)
            
            # Move the file
            shutil.copy2(source_path, target_path)
            app.logger.info(f"Saved sound sample to {target_path}")
        
        # Delete the temporary file
        os.remove(source_path)
        
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
        
        # Log the recording attempt
        app.logger.info(f"Recording attempt: user={session['username']}, class={sound_class}")
        
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
        
        # Create a temporary directory for this recording
        temp_dir = os.path.join(Config.TEMP_DIR, timestamp)
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save the original recording
        original_path = os.path.join(temp_dir, f"original_{sound_class}.webm")
        audio_data.save(original_path)
        
        # Convert audio to WAV format
        wav_path = os.path.join(temp_dir, f"original_{sound_class}.wav")
        
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(original_path)
            audio.export(wav_path, format="wav")
            app.logger.info(f"Converted audio to WAV: {wav_path}")
        except Exception as e:
            app.logger.error(f"Error converting audio: {str(e)}")
            return jsonify({
                'success': False, 
                'message': f'Error converting audio: {str(e)}'
            }), 500
        
        # Chop the recording into chunks using the SoundProcessor
        try:
            from src.audio_chunker import SoundProcessor
            processor = SoundProcessor()
            
            # Chop the recording into chunks - the SoundProcessor expects just a filename
            # not separate parameters as currently called
            chunk_files = processor.chop_recording(wav_path)
            
            # Move the resulting chunks to our timestamp directory
            # The chunks are created in the same directory as the original file
            wav_dir = os.path.dirname(wav_path)
            for chunk_file in chunk_files:
                # The chunk file is just the basename, need to construct full path
                chunk_full_path = os.path.join(wav_dir, chunk_file)
                # Make sure we prefix with the class name so the verification page works
                new_name = f"{sound_class}_{chunk_file}" if not chunk_file.startswith(sound_class) else chunk_file
                new_path = os.path.join(temp_dir, new_name)
                shutil.move(chunk_full_path, new_path)
                app.logger.info(f"Moved chunk from {chunk_full_path} to {new_path}")
            
            # Delete the original recordings to save space
            try:
                os.remove(original_path)
                os.remove(wav_path)
            except Exception as e:
                app.logger.warning(f"Could not remove original files: {str(e)}")
            
            # Check if any chunks were created
            chunks = [f for f in os.listdir(temp_dir) if f.endswith('.wav')]
            
            # Redirect to the verification page
            return redirect(url_for('verify_chunks', timestamp=timestamp))
            
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
    
    # Get all chunks for this timestamp
    chunks_dir = os.path.join(Config.TEMP_DIR, timestamp)
    if not os.path.isdir(chunks_dir):
        flash('No chunks found for verification', 'warning')
        return redirect(url_for('sounds_record'))
    
    # Get all WAV files in the directory
    chunks = []
    class_name = None
    
    for filename in os.listdir(chunks_dir):
        if filename.endswith('.wav'):
            # Extract class name from filename
            # The filename pattern should be class_name_*.wav
            if '_' in filename:
                # Use the first part as the class name
                file_class_name = filename.split('_')[0]
                if class_name is None:
                    class_name = file_class_name
                elif class_name != file_class_name:
                    app.logger.warning(f"Multiple class names detected in chunks: {class_name}, {file_class_name}")
            
            chunks.append({
                'id': os.path.join(timestamp, filename),
                'url': url_for('serve_temp_file', filename=os.path.join(timestamp, filename))
            })
    
    if not chunks:
        flash('No sound samples were detected in your recording', 'warning')
        return redirect(url_for('sounds_record'))
    
    # If we couldn't determine the class name from filenames, use a default
    if class_name is None:
        class_name = "Unknown"
        app.logger.warning(f"Could not determine class name from filenames in {chunks_dir}")
    
    # Render the verification template with the chunks
    return render_template('verify.html', chunks=chunks, class_name=class_name)

@app.route('/process_verification', methods=['POST'])
def process_verification():
    """
    Process the user's decision for a sound chunk.
    If kept, move to the class directory. If discarded, delete.
    """
    if 'username' not in session:
        return redirect(url_for('login'))
    
    chunk_id = request.form.get('chunk_id')
    is_good = request.form.get('is_good') == 'true'
    
    if not chunk_id:
        flash('No chunk ID specified', 'danger')
        return redirect(url_for('sounds_record'))
    
    # chunk_id is in format: timestamp/class_name_original_chunk_X.wav
    # or timestamp/original_chunk_X.wav (where we need to extract class_name from elsewhere)
    
    try:
        # Split by directory separator first
        parts = chunk_id.split('/')
        if len(parts) != 2:
            raise ValueError(f"Invalid chunk ID format: {chunk_id}")
            
        timestamp = parts[0]
        filename = parts[1]
        
        # Extract class name from filename
        # Pattern is either class_name_*.wav or chunk_class_name*.wav
        if '_' in filename:
            class_name = filename.split('_')[0]  # First part before underscore is the class name
        else:
            # If no underscore in filename, use the class name from the template
            class_name = request.form.get('class_name')
            if not class_name:
                raise ValueError("Could not determine class name from filename")
        
        app.logger.info(f"Processing verification for chunk: {chunk_id}, class: {class_name}, keep: {is_good}")
        
        # Get the full path to the chunk file
        chunk_path = os.path.join(Config.TEMP_DIR, chunk_id)
        
        if not os.path.exists(chunk_path):
            app.logger.error(f"Chunk file does not exist: {chunk_path}")
            return jsonify({'success': False, 'error': 'Chunk file not found'})
        
        if is_good:
            # User wants to keep this sample - move it to the appropriate class directory
            class_dir = os.path.join(Config.SOUNDS_DIR, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            # Generate a unique filename
            username = session.get('username', 'anonymous')
            timestamp_now = datetime.now().strftime('%Y%m%d%H%M%S')
            new_filename = f"{class_name}_{username}_{timestamp_now}_{uuid.uuid4().hex[:6]}.wav"
            
            # Move the file to the class directory
            new_path = os.path.join(class_dir, new_filename)
            shutil.copy2(chunk_path, new_path)
            app.logger.info(f"Saved approved chunk from {chunk_path} to {new_path}")
        
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
    file_path = os.path.join(Config.SOUNDS_DIR, class_name, sample_name)
    
    if not os.path.exists(file_path):
        return "File not found", 404
    
    return send_from_directory(Config.SOUNDS_DIR, os.path.join(class_name, sample_name))

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
if __name__ == '__main__':
    app.run(debug=Config.DEBUG, host='0.0.0.0', port=5001)