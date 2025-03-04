#!/usr/bin/env python3
"""
Model Registry Synchronization Tool

This script synchronizes the models.json registry with the model files on disk.
It scans all model folders, extracts metadata, and updates the registry to match the reality.

Usage:
    python .scripts/sync_models_registry.py [options]

Options:
    --fix-best-model    Identifies the best model for each dictionary and updates the registry
    --verbose           Display detailed information about each model
    --force-update      Force update metadata from files even if it exists in registry
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime

# Add the root directory to the Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from src.ml.model_paths import synchronize_model_registry, update_model_registry, load_model_metadata

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_sync')

def find_best_models():
    """
    Find the best model for each dictionary based on accuracy and update the registry
    """
    registry_path = os.path.join(Config.BASE_DIR, 'data', 'models', 'models.json')
    if not os.path.exists(registry_path):
        logger.error(f"Registry file not found at {registry_path}")
        return False
    
    try:
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    except Exception as e:
        logger.error(f"Error loading registry: {str(e)}")
        return False
    
    # Group models by dictionary
    models_by_dict = {}
    for model_type, models in registry.get('models', {}).items():
        for model_id, model_data in models.items():
            dict_name = model_data.get('dictionary')
            if not dict_name:
                continue
                
            if dict_name not in models_by_dict:
                models_by_dict[dict_name] = []
                
            # Ensure we have accuracy data
            accuracy = model_data.get('accuracy', 0)
            
            # If accuracy is missing, try to get it from the metadata file
            if accuracy == 0 or accuracy is None:
                metadata = load_model_metadata(model_id, model_type)
                if metadata and 'accuracy' in metadata:
                    accuracy = metadata['accuracy']
                    # Update the registry entry
                    registry['models'][model_type][model_id]['accuracy'] = accuracy
                    logger.info(f"Updated missing accuracy for {model_id}: {accuracy}")
            
            # Add model with its type and id
            models_by_dict[dict_name].append({
                'id': model_id,
                'type': model_type,
                'accuracy': accuracy,
                'created_at': model_data.get('created_at', '')
            })
    
    # Find best model for each dictionary
    best_models = {}
    for dict_name, models in models_by_dict.items():
        # Sort by accuracy (highest first), then by created_at (newest first)
        sorted_models = sorted(
            models, 
            key=lambda x: (float(x.get('accuracy', 0)), x.get('created_at', '')), 
            reverse=True
        )
        
        if sorted_models:
            best_model = sorted_models[0]
            best_models[dict_name] = {
                'id': best_model['id'],
                'type': best_model['type'],
                'accuracy': best_model['accuracy']
            }
            logger.info(f"Best model for {dict_name}: {best_model['id']} "
                        f"(accuracy: {best_model['accuracy']})")
    
    # Update registry with best models
    if "best_models" not in registry:
        registry["best_models"] = {}
        
    registry["best_models"] = {
        dict_name: model_info['id'] for dict_name, model_info in best_models.items()
    }
    
    # Mark best models in the model entries
    for model_type in registry["models"]:
        for model_id in registry["models"][model_type]:
            model_data = registry["models"][model_type][model_id]
            dict_name = model_data.get("dictionary")
            
            # Set is_best flag
            if dict_name and dict_name in best_models and best_models[dict_name]["id"] == model_id:
                model_data["is_best"] = True
            else:
                # Remove is_best flag if it exists
                if "is_best" in model_data:
                    del model_data["is_best"]
    
    # Create symlinks for best models
    for dict_name, model_info in best_models.items():
        model_id = model_info['id']
        model_type = model_info['type']
        
        # Only create symlinks for CNN models
        if model_type == 'cnn':
            best_link_path = os.path.join(Config.BASE_DIR, 'data', 'models', f'best_{dict_name}_model.h5')
            model_path = os.path.join(
                Config.BASE_DIR, 'data', 'models', 
                model_type, model_id, f"{model_id}.h5"
            )
            
            # Check if model file exists
            if not os.path.exists(model_path):
                logger.warning(f"Model file not found at {model_path}")
                continue
                
            # Create or update symlink
            try:
                # Remove existing symlink if it exists
                if os.path.exists(best_link_path):
                    os.remove(best_link_path)
                    
                # Create symlink
                os.symlink(model_path, best_link_path)
                logger.info(f"Created symlink for best {dict_name} model: {best_link_path} -> {model_path}")
                
                # Also create a generic best_cnn_model.h5 link if this is the first/best overall model
                if not os.path.exists(os.path.join(Config.BASE_DIR, 'data', 'models', 'best_cnn_model.h5')):
                    os.symlink(model_path, os.path.join(Config.BASE_DIR, 'data', 'models', 'best_cnn_model.h5'))
                    logger.info(f"Created generic symlink best_cnn_model.h5 -> {model_path}")
                
            except Exception as e:
                logger.error(f"Error creating symlink for {dict_name}: {str(e)}")
    
    # Save updated registry
    try:
        with open(registry_path, 'w') as f:
            json.dump(registry, f, indent=2)
        logger.info(f"Updated registry with best models information")
        return True
    except Exception as e:
        logger.error(f"Error saving registry: {str(e)}")
        return False

def update_missing_metadata(force_update=False):
    """
    Check for missing metadata in registry entries and update from metadata files
    """
    registry_path = os.path.join(Config.BASE_DIR, 'data', 'models', 'models.json')
    if not os.path.exists(registry_path):
        logger.error(f"Registry file not found at {registry_path}")
        return False
    
    try:
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    except Exception as e:
        logger.error(f"Error loading registry: {str(e)}")
        return False
    
    updates_made = False
    
    # Check each model for missing metadata
    for model_type, models in registry.get('models', {}).items():
        for model_id, model_data in models.items():
            # Check for missing critical fields
            needs_update = force_update
            for field in ['class_names', 'num_classes', 'accuracy', 'input_shape']:
                if field not in model_data:
                    needs_update = True
                    break
            
            if needs_update:
                # Try to get metadata from file
                metadata = load_model_metadata(model_id, model_type)
                if metadata:
                    # Update missing fields
                    fields_updated = []
                    for key, value in metadata.items():
                        if force_update or key not in model_data:
                            model_data[key] = value
                            fields_updated.append(key)
                    
                    if fields_updated:
                        updates_made = True
                        logger.info(f"Updated metadata for {model_id}: {', '.join(fields_updated)}")
    
    # Save updated registry if changes were made
    if updates_made:
        try:
            with open(registry_path, 'w') as f:
                json.dump(registry, f, indent=2)
            logger.info(f"Saved registry with updated metadata")
            return True
        except Exception as e:
            logger.error(f"Error saving registry: {str(e)}")
            return False
    else:
        logger.info("No metadata updates were needed")
        return True

def display_model_info(registry):
    """Display detailed information about models in the registry"""
    print("\n=== MODEL REGISTRY SUMMARY ===")
    
    # Count models by type
    model_counts = registry.get('counts', {})
    print(f"\nTotal models: {model_counts.get('total', 0)}")
    for model_type, count in model_counts.items():
        if model_type != 'total':
            print(f"  {model_type.upper()} models: {count}")
    
    # Show best models
    best_models = registry.get('best_models', {})
    if best_models:
        print("\nBest models by dictionary:")
        for dict_name, model_id in best_models.items():
            # Find model details
            model_type = None
            accuracy = None
            
            for mtype in registry.get('models', {}):
                if model_id in registry['models'][mtype]:
                    model_type = mtype
                    accuracy = registry['models'][mtype][model_id].get('accuracy', 'Unknown')
                    break
            
            print(f"  {dict_name}: {model_id} ({model_type.upper()}, accuracy: {accuracy})")
    
    # List models by dictionary
    models_by_dict = {}
    for model_type, models in registry.get('models', {}).items():
        for model_id, model_data in models.items():
            dict_name = model_data.get('dictionary', 'Unknown')
            if dict_name not in models_by_dict:
                models_by_dict[dict_name] = []
            models_by_dict[dict_name].append({
                'id': model_id,
                'type': model_type,
                'accuracy': model_data.get('accuracy', 'Unknown'),
                'created_at': model_data.get('created_at', 'Unknown'),
                'is_best': model_data.get('is_best', False),
                'class_names': model_data.get('class_names', []),
                'num_classes': model_data.get('num_classes', 0)
            })
    
    print("\nModels by dictionary:")
    for dict_name, models in sorted(models_by_dict.items()):
        print(f"\n  Dictionary: {dict_name} ({len(models)} models)")
        
        # Sort by creation date (newest first)
        sorted_models = sorted(
            models, 
            key=lambda x: x.get('created_at', ''), 
            reverse=True
        )
        
        for model in sorted_models:
            best_marker = "✓ (BEST)" if model.get('is_best') else ""
            class_count = f"{model['num_classes']} classes" if model['num_classes'] else ""
            class_names = f": {', '.join(model['class_names'])}" if model['class_names'] else ""
            
            print(f"    - {model['id']} ({model['type'].upper()}, "
                  f"accuracy: {model['accuracy']}) {class_count}{class_names} {best_marker}")

def main():
    parser = argparse.ArgumentParser(description="Synchronize models.json with actual model files")
    parser.add_argument('--fix-best-model', action='store_true', 
                        help="Identify the best model for each dictionary")
    parser.add_argument('--verbose', action='store_true', 
                        help="Display detailed model information")
    parser.add_argument('--force-update', action='store_true',
                        help="Force update metadata from files even if it exists in registry")
    
    args = parser.parse_args()
    
    # First synchronize the registry with filesystem
    logger.info("Starting model registry synchronization...")
    if not synchronize_model_registry():
        logger.error("Model registry synchronization failed")
        return 1
    
    logger.info("Model registry synchronization completed successfully")
    
    # Update any missing metadata
    logger.info("Checking for missing metadata...")
    if not update_missing_metadata(args.force_update):
        logger.error("Failed to update missing metadata")
        return 1
    
    # Identify best models if requested
    if args.fix_best_model:
        logger.info("Identifying best models...")
        if not find_best_models():
            logger.error("Failed to update best models")
            return 1
        logger.info("Best models updated successfully")
    
    # Display detailed info if requested
    if args.verbose:
        registry_path = os.path.join(Config.BASE_DIR, 'data', 'models', 'models.json')
        try:
            with open(registry_path, 'r') as f:
                registry = json.load(f)
            display_model_info(registry)
        except Exception as e:
            logger.error(f"Error reading registry: {str(e)}")
            return 1
    
    logger.info("All operations completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 