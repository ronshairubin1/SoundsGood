import os
import logging
from flask import jsonify, Blueprint, session
from datetime import datetime, timedelta
import random

from backend.src.services.dictionary_service import DictionaryService
from backend.config import Config
from backend.src.ml.model_paths import get_model_counts_from_registry, synchronize_model_registry, count_model_files_from_registry

# Create blueprint
dashboard_bp = Blueprint('dashboard', __name__, url_prefix='/api/dashboard')
dictionary_service = DictionaryService()

@dashboard_bp.route('/stats', methods=['GET'])
def get_dashboard_stats():
    """Get statistics for the dashboard."""
    try:
        stats = {
            'dictionaries': 0,
            'classes': 0,
            'recordings': 0,
            'models': 0
        }
        
        # Get user_id from session
        user_id = session.get('user_id')
        logging.info(f"Getting dashboard stats for user: {user_id}")
        
        # Get statistics from services
        try:
            # Count dictionaries
            dictionaries = dictionary_service.get_dictionaries(user_id)
            stats['dictionaries'] = len(dictionaries) if dictionaries else 0
            
            # Count classes across all dictionaries
            all_classes = set()
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
            
            # Count both original and augmented recordings
            total_recordings = 0
            original_recordings = 0
            augmented_recordings = 0
            
            # Count original recordings by directly checking all folders in the training directory
            # This is more reliable than using class names from dictionaries
            if os.path.exists(Config.TRAINING_SOUNDS_DIR) and os.path.isdir(Config.TRAINING_SOUNDS_DIR):
                for class_dir in os.listdir(Config.TRAINING_SOUNDS_DIR):
                    class_path = os.path.join(Config.TRAINING_SOUNDS_DIR, class_dir)
                    if os.path.isdir(class_path):
                        recordings = [f for f in os.listdir(class_path) if f.endswith('.wav') or f.endswith('.mp3')]
                        original_recordings += len(recordings)
                        logging.debug(f"Found {len(recordings)} recordings in class {class_dir}")
            
            # Count augmented recordings using a recursive approach
            # This is more reliable as augmented files might be organized differently
            def count_audio_files_recursive(directory):
                count = 0
                if not os.path.exists(directory) or not os.path.isdir(directory):
                    return 0
                    
                for root, _, files in os.walk(directory):
                    for file in files:
                        if file.endswith('.wav') or file.endswith('.mp3'):
                            count += 1
                return count
            
            # Get count from the main augmented directory
            AUGMENTED_DIR = Config.AUGMENTED_SOUNDS_DIR
            aug_count = count_audio_files_recursive(AUGMENTED_DIR)
            logging.info(f"Found {aug_count} augmented files in {AUGMENTED_DIR}")
            augmented_recordings += aug_count
            
            # Check alternative locations if needed
            ALT_AUGMENTED_DIR = os.path.join(Config.BACKEND_DATA_DIR, 'augmented_sounds')
            if os.path.exists(ALT_AUGMENTED_DIR) and os.path.isdir(ALT_AUGMENTED_DIR):
                alt_count = count_audio_files_recursive(ALT_AUGMENTED_DIR)
                logging.info(f"Found {alt_count} augmented files in alternative location {ALT_AUGMENTED_DIR}")
                augmented_recordings += alt_count
                
            # For debug purposes, let's count file extensions in the AUGMENTED_DIR
            if os.path.exists(AUGMENTED_DIR) and os.path.isdir(AUGMENTED_DIR):
                extensions = {}
                for root, _, files in os.walk(AUGMENTED_DIR):
                    for file in files:
                        ext = os.path.splitext(file)[1]
                        if ext:
                            extensions[ext] = extensions.get(ext, 0) + 1
                logging.info(f"File extensions in augmented directory: {extensions}")
            
            # Set the total recordings count
            total_recordings = original_recordings + augmented_recordings
            
            logging.info(f"Found {original_recordings} original recordings and {augmented_recordings} augmented recordings")
            
            # Add the breakdown to stats
            stats['original_recordings'] = original_recordings
            stats['augmented_recordings'] = augmented_recordings
            stats['total_recordings'] = total_recordings  # Use a different key for the template
                    
            # Keep 'recordings' for backward compatibility with any existing code
            stats['recordings'] = total_recordings
            
            # Count models directly from the models.json registry
            logging.info("Counting models from registry")
            
            # Define models_dir for use in the function call
            models_dir = os.path.join(Config.DATA_DIR, 'models')
            
            # Use the centralized function instead of the local one
            model_count, model_types_count = count_model_files_from_registry(models_dir)
            logging.info("Found %d total model files", model_count)
            logging.info("Model type breakdown - CNN: %d, RF: %d, Ensemble: %d", 
                       model_types_count['cnn'], model_types_count['rf'], model_types_count['ensemble'])
            
            stats['models'] = model_count
            stats['model_types'] = model_types_count  # Add detailed model type counts to stats
            
            logging.info("Dashboard stats before override: %s", stats)
            
            # Set the models count and log it for verification
            stats['models'] = model_count
            logging.info(f"Total model count: {model_count}")
            
            logging.info(f"Dashboard stats after override: {stats}")
        except Exception as e:
            logging.error(f"Error calculating stats: {e}")
        
        # Log the complete stats object to help with debugging
        logging.warning(f"Final stats object being returned: {stats}")
        
        # Create a response object
        response = {
            'success': True,
            'stats': stats
        }
        
        # Log the complete response payload
        logging.warning(f"Complete response payload: {response}")
        
        return jsonify(response)
    except Exception as e:
        logging.error(f"Error in get_dashboard_stats: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@dashboard_bp.route('/activity', methods=['GET'])
def get_recent_activity():
    """Get recent activity for the dashboard."""
    try:
        # Get user_id from session
        user_id = session.get('user_id')
        
        # For now, generate mock activity data
        # In a real implementation, this would come from a database
        activities = []
        
        # Try to generate some meaningful activities based on actual data
        try:
            dictionaries = dictionary_service.get_dictionaries(user_id)
            
            if dictionaries:
                # Add activity for the most recent dictionary
                newest_dict = max(dictionaries, key=lambda d: d.get('updated_at', ''))
                activities.append({
                    'type': 'dict',
                    'icon': 'folder2-open',
                    'description': f"Updated dictionary '{newest_dict['name']}'",
                    'time': _format_time(newest_dict.get('updated_at', ''))
                })
                
                # Add recording activity if there are samples
                if newest_dict.get('sample_count', 0) > 0:
                    activities.append({
                        'type': 'record',
                        'icon': 'mic-fill',
                        'description': f"Added recordings to '{newest_dict['name']}'",
                        'time': '1 hour ago'
                    })
        except Exception as e:
            logging.error(f"Error generating activity data: {e}")
        
        # Add some default activities if we don't have enough
        if len(activities) < 3:
            default_activities = [
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
            
            # Add default activities until we have at least 3
            for activity in default_activities:
                if len(activities) >= 3:
                    break
                if not any(a['description'] == activity['description'] for a in activities):
                    activities.append(activity)
        
        return jsonify({
            'success': True,
            'activities': activities
        })
    except Exception as e:
        logging.error(f"Error in get_recent_activity: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def _format_time(iso_timestamp):
    """Format a timestamp into a relative time string."""
    if not iso_timestamp:
        return 'Unknown'
    
    try:
        # Parse ISO timestamp
        dt = datetime.fromisoformat(iso_timestamp.replace('Z', '+00:00'))
        now = datetime.now()
        diff = now - dt
        
        if diff < timedelta(minutes=1):
            return 'Just now'
        elif diff < timedelta(hours=1):
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif diff < timedelta(days=1):
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif diff < timedelta(days=7):
            days = diff.days
            return f"{days} day{'s' if days != 1 else ''} ago"
        else:
            return dt.strftime('%b %d, %Y')
    except Exception:
        return 'Recently'

def format_timestamp(iso_timestamp):
    """Format a timestamp into a relative time string."""
    if not iso_timestamp:
        return 'Unknown'
    
    try:
        # Parse ISO timestamp
        dt = datetime.fromisoformat(iso_timestamp.replace('Z', '+00:00'))
        now = datetime.now()
        diff = now - dt
        
        if diff < timedelta(minutes=1):
            return 'Just now'
        elif diff < timedelta(hours=1):
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif diff < timedelta(days=1):
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif diff < timedelta(days=7):
            days = diff.days
            return f"{days} day{'s' if days != 1 else ''} ago"
        else:
            return dt.strftime('%b %d, %Y')
    except Exception:
        return 'Recently' 