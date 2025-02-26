import os
import shutil
import re
import json
import logging
from datetime import datetime
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_sound_class_dir(class_name):
    """Create a class directory in the sounds folder"""
    # Create a central sounds directory if it doesn't exist
    sounds_dir = os.path.join(Config.BASE_DIR, 'sounds')
    os.makedirs(sounds_dir, exist_ok=True)
    
    # Create the class directory
    class_path = os.path.join(sounds_dir, class_name.replace(' ', '_').lower())
    
    if not os.path.exists(class_path):
        os.makedirs(class_path, exist_ok=True)
        logging.info(f"Created class directory: {class_path}")
        return True
    else:
        logging.info(f"Class directory already exists: {class_path}")
        return False

def extract_class_from_filename(filename):
    """Extract the class name from a filename - assumes filename starts with class name"""
    # This regex matches everything up to the first underscore or digit
    match = re.match(r'([a-zA-Z]+)', filename)
    if match:
        return match.group(1).lower()
    return None

def create_dictionary_with_references(dict_name, class_names, description=""):
    """Create a dictionary that references the specified classes"""
    try:
        from src.services.dictionary_service import DictionaryService
        service = DictionaryService()
        
        # Check if dictionary exists
        safe_dict_name = dict_name.replace(' ', '_').lower()
        if safe_dict_name in service.metadata.get("dictionaries", {}):
            logging.info(f"Dictionary '{dict_name}' already exists, updating it")
            dict_info = service.metadata["dictionaries"][safe_dict_name]
        else:
            # Create the dictionary
            result = service.create_dictionary(dict_name, description)
            if not result.get("success"):
                logging.error(f"Failed to create dictionary: {result.get('error')}")
                return False
            dict_info = service.metadata["dictionaries"][safe_dict_name]
        
        # Update the classes
        sample_count = 0
        for class_name in class_names:
            safe_class_name = class_name.replace(' ', '_').lower()
            if safe_class_name not in dict_info["classes"]:
                dict_info["classes"].append(safe_class_name)
                
                # Count samples in this class
                class_path = os.path.join(Config.BASE_DIR, 'sounds', safe_class_name)
                if os.path.exists(class_path):
                    wav_files = [f for f in os.listdir(class_path) if f.lower().endswith('.wav')]
                    sample_count += len(wav_files)
        
        # Update sample count and timestamp
        dict_info["sample_count"] = sample_count
        dict_info["updated_at"] = datetime.now().isoformat()
        
        # Save the metadata
        service._save_metadata()
        logging.info(f"Updated dictionary '{dict_name}' with classes: {', '.join(class_names)}")
        return True
    except Exception as e:
        logging.error(f"Error creating/updating dictionary: {e}")
        return False

def create_example_dictionaries(available_classes):
    """Create example dictionaries using the available classes"""
    # Only proceed if we have classes
    if not available_classes:
        logging.warning("No classes available to create example dictionaries")
        return
    
    # Create dictionaries based on available classes
    class_list = list(available_classes)
    
    # Create a dictionary with all classes
    create_dictionary_with_references(
        "All Classes", 
        class_list,
        "Contains all available sound classes"
    )
    
    # Create a two-word dictionary if possible
    if len(class_list) >= 2:
        create_dictionary_with_references(
            "Two Words", 
            class_list[:2],
            "Dictionary with two sound classes"
        )
    
    # Create a three-word dictionary if possible
    if len(class_list) >= 3:
        create_dictionary_with_references(
            "Three Words", 
            class_list[:3],
            "Dictionary with three sound classes"
        )

def migrate_sounds():
    """Migrate sounds from old locations to central class directories"""
    # Define source directories
    source_dirs = [
        os.path.join(Config.BASE_DIR, 'static', 'goodsounds'),
        os.path.join(Config.BASE_DIR, 'src', 'static', 'goodsounds'),
        os.path.join(Config.BASE_DIR, 'src', 'static', 'ah')
    ]
    
    # Track statistics
    stats = {
        "total_files": 0,
        "migrated_files": 0,
        "classes_created": 0,
        "classes": set()
    }
    
    # Process each source directory
    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            logging.info(f"Source directory does not exist: {source_dir}")
            continue
        
        logging.info(f"Processing directory: {source_dir}")
        
        # Check if the directory contains class subdirectories
        subdirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
        
        if subdirs:
            # This directory has subdirectories - each likely represents a class
            for subdir in subdirs:
                class_name = subdir.lower()
                stats["classes"].add(class_name)
                
                # Create class directory
                if create_sound_class_dir(class_name):
                    stats["classes_created"] += 1
                
                # Process all wav files in the subdirectory
                subdir_path = os.path.join(source_dir, subdir)
                wav_files = [f for f in os.listdir(subdir_path) if f.lower().endswith('.wav')]
                
                for wav_file in wav_files:
                    stats["total_files"] += 1
                    source_file = os.path.join(subdir_path, wav_file)
                    dest_file = os.path.join(
                        Config.BASE_DIR, 
                        'sounds',
                        class_name.replace(' ', '_').lower(),
                        wav_file
                    )
                    
                    # Copy the file
                    try:
                        shutil.copy2(source_file, dest_file)
                        logging.info(f"Migrated: {source_file} -> {dest_file}")
                        stats["migrated_files"] += 1
                    except Exception as e:
                        logging.error(f"Failed to copy {source_file}: {e}")
        else:
            # This directory doesn't have subdirectories - files are likely directly in it
            wav_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.wav')]
            
            for wav_file in wav_files:
                stats["total_files"] += 1
                
                # Try to extract class name from the filename
                class_name = extract_class_from_filename(wav_file)
                
                if not class_name:
                    class_name = "unknown"
                
                stats["classes"].add(class_name)
                
                # Create class directory
                if create_sound_class_dir(class_name):
                    stats["classes_created"] += 1
                
                source_file = os.path.join(source_dir, wav_file)
                dest_file = os.path.join(
                    Config.BASE_DIR, 
                    'sounds',
                    class_name.replace(' ', '_').lower(),
                    wav_file
                )
                
                # Copy the file
                try:
                    shutil.copy2(source_file, dest_file)
                    logging.info(f"Migrated: {source_file} -> {dest_file}")
                    stats["migrated_files"] += 1
                except Exception as e:
                    logging.error(f"Failed to copy {source_file}: {e}")
    
    # Create example dictionaries that reference these classes
    create_example_dictionaries(stats["classes"])
    
    # Print summary
    logging.info("\nMigration Summary:")
    logging.info(f"Total files processed: {stats['total_files']}")
    logging.info(f"Files migrated: {stats['migrated_files']}")
    logging.info(f"Sound classes created: {stats['classes_created']}")
    logging.info(f"Classes: {', '.join(stats['classes'])}")
    
    # Instructions for next steps
    logging.info("\nNext Steps:")
    logging.info("1. Update Config.py to include SOUNDS_DIR = os.path.join(BASE_DIR, 'sounds')")
    logging.info("2. Update DictionaryService to reference sounds from central location")
    logging.info("3. Verify dictionaries reference the correct sound classes")
    
    return stats

if __name__ == "__main__":
    logging.info("Starting sound file migration...")
    stats = migrate_sounds()
    logging.info("Migration complete!") 