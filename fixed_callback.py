def prediction_callback(prediction):
    """
    Callback function for updating inference statistics.
    This is called after each prediction to update the stats.
    
    Args:
        prediction (dict): Prediction result
    """
    global LATEST_PREDICTION
    LATEST_PREDICTION = prediction
    logging.info(f"Got prediction: {prediction}")
    
    # Import Flask components here to avoid circular imports
    from flask import current_app
    from flask import _app_ctx_stack
    
    # We need to check if we have an application context
    if _app_ctx_stack.top is None:
        # If we're called from outside a request context, we can't access current_app
        logging.warning("prediction_callback called outside application context")
        # Store prediction but don't try to update stats
        return
    
    try:
        # Check if inference_stats exists directly on the app
        if hasattr(current_app, 'inference_stats'):
            stats = current_app.inference_stats
        # Check if it exists on the ml_api object
        elif hasattr(current_app, 'ml_api') and hasattr(current_app.ml_api, 'inference_service'):
            stats = current_app.ml_api.inference_service.inference_stats
        # As a fallback, create a new stats object if needed
        else:
            logging.warning("Creating new inference_stats as it wasn't found on the app")
            if not hasattr(current_app, 'inference_stats'):
                current_app.inference_stats = {
                    'total_predictions': 0,
                    'class_counts': {},
                    'confidence_levels': [],
                    'confusion_matrix': {}
                }
            stats = current_app.inference_stats
        
        # Update statistics
        stats['total_predictions'] = stats.get('total_predictions', 0) + 1
        c = prediction['prediction']['class']
        conf = prediction['prediction']['confidence']
        
        # Update class counts
        stats['class_counts'].setdefault(c, 0)
        stats['class_counts'][c] += 1
        stats['confidence_levels'] = stats.get('confidence_levels', []) + [conf]
        
        # Initialize confusion matrix if needed
        if 'confusion_matrix' not in stats:
            stats['confusion_matrix'] = {}
    except Exception as e:
        logging.error(f"Error updating inference stats: {e}")
        import traceback
        logging.error(traceback.format_exc())
 