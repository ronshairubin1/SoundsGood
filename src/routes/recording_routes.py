#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Route handlers for audio recording and training data collection.

These routes provide endpoints for:
- Recording training audio
- Managing training data
- Recording audio for inference
"""

import os
import json
import logging
import numpy as np
from flask import Blueprint, request, jsonify, render_template
from src.services.recording_service import RecordingService
import soundfile as sf
import tempfile

# Create Blueprint
recording_bp = Blueprint('recording', __name__)

# Initialize recording service
recording_service = RecordingService()

@recording_bp.route('/record_training', methods=['GET'])
def record_training_view():
    """Render the training data recording page."""
    # Get available classes
    classes = recording_service.get_available_classes()
    
    # Count samples per class
    class_counts = {}
    for cls in classes:
        class_counts[cls] = recording_service.get_class_sample_count(cls)
    
    return render_template('record_training.html', 
                          classes=classes,
                          class_counts=class_counts)

@recording_bp.route('/api/recording/classes', methods=['GET'])
def get_classes():
    """Get available sound classes."""
    classes = recording_service.get_available_classes()
    
    # Get count for each class
    class_data = []
    for cls in classes:
        count = recording_service.get_class_sample_count(cls)
        class_data.append({
            'name': cls,
            'count': count
        })
    
    return jsonify({
        'status': 'success',
        'classes': class_data
    })

@recording_bp.route('/api/recording/calibrate', methods=['POST'])
def calibrate_microphone():
    """Calibrate microphone for ambient noise."""
    duration = request.json.get('duration', 1.0)
    
    try:
        noise_level = recording_service.calibrate_microphone(duration)
        
        return jsonify({
            'status': 'success',
            'noise_level': noise_level
        })
    except Exception as e:
        logging.error(f"Error calibrating microphone: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@recording_bp.route('/api/recording/record_class', methods=['POST'])
def record_class():
    """Record training data for a sound class."""
    class_name = request.json.get('class')
    max_duration = float(request.json.get('max_duration', 15.0))
    
    if not class_name:
        return jsonify({
            'status': 'error',
            'message': 'Class name is required'
        }), 400
    
    # Define event stream callback
    response_data = {'status': 'processing', 'progress': 0, 'message': ''}
    
    def callback(progress, message):
        nonlocal response_data
        response_data['progress'] = progress
        response_data['message'] = message
    
    # Start recording process
    result = recording_service.record_training_sounds(
        class_name, 
        max_duration=max_duration,
        callback=callback
    )
    
    # Update response data with results
    response_data.update({
        'status': 'success' if result.get('original_count', 0) > 0 else 'error',
        'class': class_name,
        'original_count': result.get('original_count', 0),
        'augmented_count': result.get('augmented_count', 0),
        'total_count': result.get('total_count', 0)
    })
    
    if 'error' in result:
        response_data['message'] = result['error']
        response_data['status'] = 'error'
    
    return jsonify(response_data)

@recording_bp.route('/api/recording/inference', methods=['POST'])
def record_for_inference():
    """Record audio for model inference."""
    max_duration = float(request.json.get('max_duration', 5.0))
    
    # Define event stream callback
    response_data = {'status': 'processing', 'progress': 0, 'message': ''}
    
    def callback(progress, message):
        nonlocal response_data
        response_data['progress'] = progress
        response_data['message'] = message
    
    # Record and process audio
    audio_data = recording_service.record_for_inference(
        max_duration=max_duration,
        callback=callback
    )
    
    if audio_data is None:
        return jsonify({
            'status': 'error',
            'message': 'No valid audio recorded'
        }), 400
    
    # Save audio to a temporary file
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name
            
        sf.write(temp_path, audio_data, recording_service.sample_rate)
        
        return jsonify({
            'status': 'success',
            'message': 'Audio recorded successfully',
            'audio_path': temp_path
        })
    except Exception as e:
        logging.error(f"Error saving recorded audio: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Error saving audio: {str(e)}'
        }), 500

@recording_bp.route('/api/recording/new_class', methods=['POST'])
def create_class():
    """Create a new sound class."""
    class_name = request.json.get('class')
    
    if not class_name:
        return jsonify({
            'status': 'error',
            'message': 'Class name is required'
        }), 400
    
    # Create class directory
    try:
        class_dir = os.path.join(str(recording_service.training_dir), class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        return jsonify({
            'status': 'success',
            'message': f'Created class "{class_name}"',
            'class': class_name
        })
    except Exception as e:
        logging.error(f"Error creating class directory: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@recording_bp.route('/api/recording/approve_sound', methods=['POST'])
def approve_sound():
    """Approve a recorded sound and save it for training."""
    class_name = request.json.get('class')
    audio_path = request.json.get('audio_path')
    is_approved = request.json.get('approved', True)
    
    if not class_name or not audio_path:
        return jsonify({
            'status': 'error',
            'message': 'Class name and audio path are required'
        }), 400
    
    try:
        # Load audio data
        audio_data, _ = sf.read(audio_path)
        
        # Save to training directory if approved
        if is_approved:
            metadata = {
                'approved': True,
                'original': True
            }
            
            saved_path = recording_service.save_training_sound(
                audio_data,
                class_name,
                is_approved=True,
                metadata=metadata
            )
            
            if saved_path:
                return jsonify({
                    'status': 'success',
                    'message': f'Sound saved for class "{class_name}"',
                    'path': saved_path
                })
            else:
                return jsonify({
                    'status': 'error',
                    'message': 'Failed to save sound'
                }), 500
        else:
            return jsonify({
                'status': 'success',
                'message': 'Sound discarded'
            })
    
    except Exception as e:
        logging.error(f"Error approving sound: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
    finally:
        # Clean up the temporary file
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except:
            pass 