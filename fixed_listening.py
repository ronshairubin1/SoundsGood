@ml_bp.route('/start_listening', methods=['POST'])
def start_listening_api():
    """
    Start a sound detector listening to the microphone.
    This will create a sound detector instance and keep it running in the background.
    
    Returns:
        JSON response with status
    """
    try:
        # Get parameters from request
        dict_name = request.form.get('dict_name', '')
        use_ambient_noise = request.form.get('use_ambient_noise', 'false').lower() == 'true'
        
        # Determine model path
        if dict_name:
            model_path = os.path.join('models', f"{dict_name.replace(' ', '_')}_model.h5")
        else:
            # If no dictionary name provided, use the active dictionary
            active_dict = Config.get_dictionary()
            dict_name = active_dict['name']
            model_path = os.path.join('models', f"{dict_name.replace(' ', '_')}_model.h5")
        
        # Check if model exists
        if not os.path.exists(model_path):
            return jsonify({
                'status': 'error',
                'message': f"Model not found for dictionary: {dict_name}"
            })
            
        # Get class names
        active_dict = Config.get_dictionary()
        class_names = active_dict['sounds']
            
        # Load the model
        try:
            logging.info("Loading Keras model...")
            with tf.keras.utils.custom_object_scope({'BatchShape': lambda x: None}):
                model = tf.keras.models.load_model(model_path)
                logging.info(f"Model loaded successfully with input shape: {model.input_shape}")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            traceback.print_exc()
            return jsonify({
                'status': 'error', 
                'message': f"Error loading model: {str(e)}"
            })
            
        # Create and start the sound detector
        try:
            logging.info("Creating SoundDetector...")
            sound_detector = SoundDetector(model, class_names, use_ambient_noise=use_ambient_noise)
            logging.info("SoundDetector created successfully")
            
            logging.info("Starting listening...")
            sound_detector.start_listening(callback=prediction_callback)
            logging.info("Listening started successfully")
            
            return jsonify({
                'status': 'success', 
                'message': 'Started listening', 
                'sound_classes': class_names
            })
        except Exception as e:
            logging.error(f"Error creating/starting SoundDetector: {e}")
            traceback.print_exc()
            return jsonify({
                'status': 'error', 
                'message': f"Error starting listener: {str(e)}"
            })
            
    except Exception as e:
        logging.error(f"Error in start_listening_api: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f"Unexpected error: {str(e)}"
        }) 