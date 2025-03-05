@ml_bp.route('/model_metadata/<model_id>', methods=['GET'])
def get_model_metadata_direct(model_id):
    """Direct endpoint for model metadata"""
    try:
        # Get model info from models.json
        models_json_path = os.path.join("models", "models.json")
        if not os.path.exists(models_json_path):
            return jsonify({
                'status': 'error',
                'message': 'Models registry not found'
            })
            
        with open(models_json_path, 'r') as f:
            registry = json.load(f)
            
        # Check if model exists in registry
        model_types = registry.get('models', {})
        model_data = None
        
        for model_type, models in model_types.items():
            if model_id in models:
                model_data = models[model_id]
                model_data['type'] = model_type
                break
                
        if not model_data:
            return jsonify({
                'status': 'error',
                'message': f'Model {model_id} not found in registry'
            })
            
        # Check for metadata file
        metadata_path = os.path.join("models", "metadata", f"{model_id}.json")
        metadata = {}
        
        if os.path.exists(metadata_path):
            print(f"DEBUG: Metadata file exists, loading content")
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                print(f"DEBUG: Metadata keys: {list(metadata.keys())}")
            except Exception as file_error:
                print(f"DEBUG: Error reading metadata file: {str(file_error)}")
                return jsonify({
                    'status': 'error',
                    'message': f'Error reading metadata file: {str(file_error)}'
                })
                
        # Combine model data with metadata
        model_data.update(metadata)
        
        return jsonify({
            'status': 'success',
            'model_id': model_id,
            'metadata': metadata
        })
        
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f'Error retrieving model metadata: {str(e)}'
        }) 