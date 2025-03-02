import os
import logging
import json
from flask import request, jsonify, Blueprint, current_app, session
from datetime import datetime

from src.services.dictionary_service import DictionaryService
from config import Config

# Create blueprint
dictionary_bp = Blueprint('dictionary', __name__, url_prefix='/api/dictionary')
dictionary_service = DictionaryService()

@dictionary_bp.route('/list', methods=['GET'])
def list_dictionaries():
    """List all dictionaries."""
    # Get user_id from query params or session
    user_id = request.args.get('user_id')
    
    # If no user_id provided, try to get from session
    if not user_id and 'user_id' in session:
        user_id = session.get('user_id')
        logging.debug(f"Using user_id from session: {user_id}")
    elif not user_id:
        # For debugging, use default or list all
        user_id = None  # None will list all dictionaries
        logging.debug("No user_id provided, will list all dictionaries")
    
    logging.debug(f"Listing dictionaries for user_id: {user_id}")
    
    try:
        dictionaries = dictionary_service.get_dictionaries(user_id)
        logging.debug(f"Found {len(dictionaries)} dictionaries")
        
        return jsonify({
            'success': True,
            'dictionaries': dictionaries
        })
    except Exception as e:
        logging.error(f"Error listing dictionaries: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@dictionary_bp.route('/create', methods=['POST'])
def create_dictionary():
    """Create a new dictionary."""
    try:
        # Enhanced logging to debug request issues
        logging.info(f"Create dictionary request received")
        logging.info(f"Request method: {request.method}")
        logging.info(f"Request content type: {request.content_type}")
        
        # Get data from either form submission or JSON
        if request.content_type and 'application/json' in request.content_type:
            # Handle JSON request
            try:
                data = request.get_json(force=True, silent=True)
                logging.info(f"Received JSON data: {data}")
            except Exception as json_err:
                logging.error(f"Error parsing JSON: {json_err}")
                return jsonify({
                    'success': False,
                    'error': f'Invalid JSON: {str(json_err)}'
                }), 400
        else:
            # Handle form submission
            data = {}
            data['name'] = request.form.get('name', '')
            data['description'] = request.form.get('description', '')
            redirect_url = request.form.get('redirect', '/dictionaries/manage')
            logging.info(f"Received form data: {data}")
            logging.info(f"Redirect URL: {redirect_url}")
        
        # Debug logging for session info
        logging.info(f"Session data: {session}")
        
        if not data or not data.get('name'):
            logging.error("Missing dictionary name in request")
            error_msg = 'Dictionary name is required'
            
            # Check if this was a form submission
            if request.form:
                from flask import redirect, url_for, flash
                flash(error_msg, 'danger')
                return redirect(redirect_url)
            
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        
        # Ensure user_id is passed
        user_id = data.get('user_id')
        if not user_id and 'user_id' in session:
            user_id = session.get('user_id')
            logging.info(f"Using user_id from session: {user_id}")
        elif not user_id:
            # For debugging purposes, we'll use a default user_id
            user_id = "default_user"
            logging.info(f"No user_id provided, using default: {user_id}")
        
        # Check if a dictionary with this name already exists
        safe_name = data['name'].replace(' ', '_').lower()
        logging.info(f"Checking if dictionary '{safe_name}' already exists")
        
        if safe_name in dictionary_service.metadata.get("dictionaries", {}):
            logging.warning(f"Dictionary '{data['name']}' already exists")
            error_msg = f"Dictionary '{data['name']}' already exists"
            
            # Check if this was a form submission
            if request.form:
                from flask import redirect, url_for, flash
                flash(error_msg, 'danger')
                return redirect(redirect_url)
            
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
            
        logging.info(f"Creating new dictionary '{data['name']}' for user {user_id}")
        result = dictionary_service.create_dictionary(
            data['name'], 
            data.get('description', ''),
            user_id
        )
        
        logging.info(f"Dictionary creation result: {result}")
        
        if result['success']:
            logging.info(f"Dictionary '{data['name']}' created successfully")
            
            # Check if this was a form submission
            if request.form:
                from flask import redirect, url_for, flash
                flash(f"Dictionary '{data['name']}' created successfully!", 'success')
                return redirect(redirect_url)
            
            return jsonify(result)
        else:
            logging.error(f"Failed to create dictionary: {result.get('error')}")
            error_msg = result.get('error', 'Failed to create dictionary')
            
            # Check if this was a form submission
            if request.form:
                from flask import redirect, url_for, flash
                flash(error_msg, 'danger')
                return redirect(redirect_url)
            
            return jsonify(result), 400
    except Exception as e:
        logging.exception(f"Error creating dictionary: {e}")
        error_msg = f"Server error: {str(e)}"
        
        # Check if this was a form submission
        if request.form:
            from flask import redirect, url_for, flash
            flash(error_msg, 'danger')
            return redirect(request.form.get('redirect', '/dictionaries/manage'))
        
        return jsonify({
            'success': False,
            'error': error_msg
        }), 500

@dictionary_bp.route('/<dict_name>/add_class', methods=['POST'])
def add_class(dict_name):
    """Add a class to a dictionary."""
    logging.debug(f"add_class called with dict_name: {dict_name}")
    data = request.get_json()
    logging.debug(f"add_class request data: {data}")
    
    if not data or 'class_name' not in data:
        logging.error("Missing class_name in request data")
        return jsonify({
            'success': False,
            'error': 'Missing class name'
        }), 400
    
    try:
        result = dictionary_service.add_class(dict_name, data['class_name'])
        logging.debug(f"add_class result: {result}")
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400
    except Exception as e:
        logging.error(f"Error in add_class: {e}")
        return jsonify({
            'success': False,
            'error': f"Server error: {str(e)}"
        }), 500

@dictionary_bp.route('/<dict_name>/<class_name>/add_sample', methods=['POST'])
def add_sample(dict_name, class_name):
    """Add a sample to a class."""
    if 'file' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No file provided'
        }), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'Empty filename'
        }), 400
    
    # Save file temporarily
    temp_path = os.path.join(Config.TEMP_DIR, file.filename)
    file.save(temp_path)
    
    try:
        # Check if we need to create a new class
        create_class = request.form.get('create_class') == 'true'
        
        # If class doesn't exist and create_class flag is true, create it
        sounds_dir = Config.SOUNDS_DIR
        class_path = os.path.join(sounds_dir, class_name)
        
        if create_class and not os.path.exists(class_path):
            logging.info(f"Creating new sound class directory: {class_path}")
            os.makedirs(class_path, exist_ok=True)
        elif not os.path.exists(class_path):
            # Class doesn't exist and we're not supposed to create it
            raise ValueError(f"Sound class '{class_name}' does not exist")
        
        # Add to dictionary
        result = dictionary_service.add_sample(
            dict_name, 
            class_name, 
            temp_path,
            request.form.get('sample_name')
        )
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
            
        return jsonify(result)
    except Exception as e:
        logging.error(f"Error adding sample: {e}")
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@dictionary_bp.route('/<dict_name>/<class_name>/samples', methods=['GET'])
def get_samples(dict_name, class_name):
    """Get all samples for a class."""
    logging.debug(f"Getting samples for class '{class_name}' in dictionary '{dict_name}'")
    
    result = dictionary_service.get_samples(dict_name, class_name)
    
    if result['success']:
        return jsonify(result)
    else:
        return jsonify(result), 400

@dictionary_bp.route('/<dict_name>', methods=['DELETE'])
def delete_dictionary(dict_name):
    """Delete a dictionary."""
    result = dictionary_service.delete_dictionary(dict_name)
    
    if result['success']:
        return jsonify(result)
    else:
        return jsonify(result), 400

@dictionary_bp.route('/debug', methods=['GET'])
def debug_metadata():
    """Debug endpoint to view metadata directly."""
    # Only allow in development/debug mode
    if not Config.DEBUG:
        return jsonify({
            'success': False,
            'error': 'Debug endpoint only available in DEBUG mode'
        }), 403
    
    try:
        metadata_file = os.path.join(dictionary_service.dictionaries_dir, 'metadata.json')
        logging.debug(f"Reading metadata from {metadata_file}")
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Get file details
            file_size = os.path.getsize(metadata_file)
            file_modified = os.path.getmtime(metadata_file)
            
            return jsonify({
                'success': True,
                'metadata': metadata,
                'file_info': {
                    'path': metadata_file,
                    'size': file_size,
                    'last_modified': datetime.fromtimestamp(file_modified).isoformat(),
                    'exists': True
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Metadata file does not exist',
                'file_info': {
                    'path': metadata_file,
                    'exists': False
                }
            })
    except Exception as e:
        logging.error(f"Error in debug endpoint: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@dictionary_bp.route('/<dict_name>/<class_name>', methods=['DELETE'])
def delete_class(dict_name, class_name):
    """Delete a class from a dictionary."""
    logging.debug(f"Deleting class '{class_name}' from dictionary '{dict_name}'")
    
    result = dictionary_service.delete_class(dict_name, class_name)
    
    if result['success']:
        return jsonify(result)
    else:
        return jsonify(result), 400

@dictionary_bp.route('/sync_counts', methods=['POST'])
def sync_sample_counts():
    """Sync all dictionary sample counts with disk."""
    # Only allow in development/debug mode
    if not Config.DEBUG:
        return jsonify({
            'success': False,
            'error': 'This endpoint is only available in DEBUG mode'
        }), 403
    
    try:
        # Get all dictionaries
        dictionaries = dictionary_service.get_dictionaries()
        
        # Log the pre-sync information
        logging.debug("Pre-sync dictionary counts:")
        for dict_info in dictionaries:
            logging.debug(f"Dictionary '{dict_info['name']}': {dict_info['sample_count']} samples")
        
        # Create a sync summary
        sync_results = {
            'dictionaries_synced': 0,
            'details': []
        }
        
        # Force sync each dictionary
        for dict_info in dictionaries:
            dict_name = dict_info['name']
            pre_count = dict_info['sample_count']
            
            # Sync the dictionary
            dictionary_service.sync_dictionary_samples(dict_name)
            
            # Get updated info
            updated_dict = dictionary_service.get_dictionary(dict_name)
            post_count = updated_dict['sample_count']
            
            # Add to results
            sync_results['dictionaries_synced'] += 1
            sync_results['details'].append({
                'name': dict_name,
                'pre_sync_count': pre_count,
                'post_sync_count': post_count,
                'difference': post_count - pre_count
            })
            
            logging.debug(f"Synced dictionary '{dict_name}': {pre_count} → {post_count} samples")
        
        return jsonify({
            'success': True,
            'message': f"Successfully synced {sync_results['dictionaries_synced']} dictionaries",
            'results': sync_results
        })
    except Exception as e:
        logging.error(f"Error syncing sample counts: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@dictionary_bp.route('/<dict_name>/update', methods=['PUT'])
def update_dictionary(dict_name):
    """Update a dictionary's information."""
    try:
        data = request.get_json()
        
        if not data:
            logging.error("No JSON data provided in request")
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
            
        # Get required fields
        new_name = data.get('name')
        new_description = data.get('description', '')
        
        if not new_name:
            logging.error("Missing dictionary name in request")
            return jsonify({
                'success': False,
                'error': 'Missing dictionary name'
            }), 400
        
        # Get the current dictionary
        safe_dict_name = dict_name.replace(' ', '_').lower()
        dictionary = dictionary_service.metadata["dictionaries"].get(safe_dict_name)
        
        if not dictionary:
            logging.error(f"Dictionary '{dict_name}' not found for update")
            return jsonify({
                'success': False,
                'error': f"Dictionary '{dict_name}' not found"
            }), 404
        
        # Update the dictionary
        dictionary['name'] = new_name
        dictionary['description'] = new_description
        dictionary['updated_at'] = datetime.now().isoformat()
        
        # If name changed, update the dictionary key
        new_safe_name = new_name.replace(' ', '_').lower()
        if new_safe_name != safe_dict_name:
            # Create new entry with updated name
            dictionary_service.metadata["dictionaries"][new_safe_name] = dictionary
            # Remove old entry
            del dictionary_service.metadata["dictionaries"][safe_dict_name]
            
            # Move directory if it exists
            old_path = os.path.join(dictionary_service.dictionaries_dir, safe_dict_name)
            new_path = os.path.join(dictionary_service.dictionaries_dir, new_safe_name)
            if os.path.exists(old_path):
                try:
                    os.makedirs(os.path.dirname(new_path), exist_ok=True)
                    os.rename(old_path, new_path)
                    dictionary['path'] = new_path
                except Exception as e:
                    logging.error(f"Error moving dictionary directory: {e}")
        
        # Save the updated metadata
        if dictionary_service._save_metadata():
            return jsonify({
                'success': True,
                'dictionary': dictionary
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to save dictionary metadata'
            }), 500
    
    except Exception as e:
        logging.exception(f"Error updating dictionary: {e}")
        return jsonify({
            'success': False,
            'error': f"Server error: {str(e)}"
        }), 500
