"""
Routes related to feature management and processing.

This module provides API endpoints for resetting the feature cache and managing feature extraction.
"""

import os
import logging
import traceback
from flask import Blueprint, request, jsonify, current_app

# Import services
from src.services.training_service import TrainingService

# Create a blueprint for feature-related routes
feature_routes = Blueprint('feature_routes', __name__)

@feature_routes.route('/api/reset-feature-cache', methods=['POST'])
def reset_feature_cache():
    """
    Reset the feature cache to force reprocessing of audio files.
    This is useful when the feature extraction logic has been updated.
    
    Returns:
        JSON with results of the cache clearing operation
    """
    try:
        training_service = TrainingService()
        result = training_service.clear_feature_cache()
        
        logging.info(f"Feature cache reset: {result}")
        return jsonify({
            'success': result['success'],
            'message': result['message'],
            'cleared_files': result['cleared']
        })
    except Exception as e:
        error_msg = f"Error resetting feature cache: {str(e)}"
        logging.error(error_msg)
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'message': error_msg,
            'cleared_files': 0
        }), 500
