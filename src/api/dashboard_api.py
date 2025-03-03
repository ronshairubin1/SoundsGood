import os
import logging
from flask import jsonify, Blueprint, session
from datetime import datetime, timedelta
import random

from src.services.dictionary_service import DictionaryService
from config import Config

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
            
            # Count actual wav files in the sounds directory
            total_recordings = 0
            for class_name in all_classes:
                class_path = os.path.join(Config.TRAINING_SOUNDS_DIR, class_name)
                if os.path.exists(class_path) and os.path.isdir(class_path):
                    recordings = [f for f in os.listdir(class_path) if f.endswith('.wav') or f.endswith('.mp3')]
                    total_recordings += len(recordings)
                    
            stats['recordings'] = total_recordings
            
            # Count models (by checking files in models directory)
            model_files = [f for f in os.listdir(Config.MODELS_DIR) 
                          if f.endswith('.h5') or f.endswith('.joblib')]
            stats['models'] = len(model_files)
            
            logging.info(f"Dashboard stats: {stats}")
        except Exception as e:
            logging.error(f"Error calculating stats: {e}")
        
        return jsonify({
            'success': True,
            'stats': stats
        })
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