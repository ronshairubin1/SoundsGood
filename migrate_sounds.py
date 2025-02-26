import os
import shutil
import re
import logging
from datetime import datetime
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_dictionary_if_not_exists(dict_name, dict_description="Migrated from goodsounds"):
    """Create a dictionary if it doesn't exist"""
    dict_path = os.path.join(Config.DICTIONARIES_DIR, dict_name.replace(' ', '_').lower())
    
    if not os.path.exists(dict_path):
        os.makedirs(dict_path, exist_ok=True)
        logging.info(f"Created dictionary directory: {dict_path}")
        return True
    else:
        logging.info(f"Dictionary already exists: {dict_path}")
        return False

def create_class_if_not_exists(dict_name, class_name):
    """Create a class within a dictionary if it doesn't exist"""
    dict_path = os.path.join(Config.DICTIONARIES_DIR, dict_name.replace(' ', '_').lower())
    class_path = os.path.join(dict_path, class_name.replace(' ', '_').lower())
    
    if not os.path.exists(class_path):
        os.makedirs(class_path, exist_ok=True)
        logging.info(f"Created class directory: {class_path}")
        return True
    else:
        logging.info(f"Class already exists: {class_path}")
        return False

def extract_class_from_filename(filename):
    """Extract the class name from a filename - assumes filename starts with class name"""
    # This regex matches everything up to the first underscore or digit
    match = re.match(r'([a-zA-Z]+)', filename)
    if match:
        return match.group(1).lower()
    return None

def migrate_sounds():
    """Migrate sounds from old location to new dictionary structure"""
    # Define source directories
    source_dirs = [
        os.path.join(Config.CURRENT_DIR, 'static', 'goodsounds'),
        os.path.join(Config.CURRENT_DIR, 'src', 'static', 'goodsounds'),
        os.path.join(Config.CURRENT_DIR, 'src', 'static', 'ah')
    ]
    
    # Create a main dictionary for migrated sounds
    main_dict_name = "migrated_sounds"
    create_dictionary_if_not_exists(main_dict_name, "Sounds migrated from legacy structure")
    
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
                
                # Create class in the new structure
                if create_class_if_not_exists(main_dict_name, class_name):
                    stats["classes_created"] += 1
                
                # Process all wav files in the subdirectory
                subdir_path = os.path.join(source_dir, subdir)
                wav_files = [f for f in os.listdir(subdir_path) if f.lower().endswith('.wav')]
                
                for wav_file in wav_files:
                    stats["total_files"] += 1
                    source_file = os.path.join(subdir_path, wav_file)
                    dest_file = os.path.join(
                        Config.DICTIONARIES_DIR, 
                        main_dict_name.replace(' ', '_').lower(),
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
                
                # Create class in the new structure
                if create_class_if_not_exists(main_dict_name, class_name):
                    stats["classes_created"] += 1
                
                source_file = os.path.join(source_dir, wav_file)
                dest_file = os.path.join(
                    Config.DICTIONARIES_DIR, 
                    main_dict_name.replace(' ', '_').lower(),
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
    
    # Update dictionary metadata
    try:
        from src.services.dictionary_service import DictionaryService
        service = DictionaryService()
        
        # Force reload metadata to ensure it's up-to-date
        service._load_metadata()
        
        # Ensure the dictionary exists in metadata
        if main_dict_name.replace(' ', '_').lower() not in service.metadata["dictionaries"]:
            service.create_dictionary(main_dict_name, "Sounds migrated from legacy structure")
        
        # Update class list and sample count
        dict_info = service.metadata["dictionaries"][main_dict_name.replace(' ', '_').lower()]
        for class_name in stats["classes"]:
            safe_class_name = class_name.replace(' ', '_').lower()
            if safe_class_name not in dict_info["classes"]:
                dict_info["classes"].append(safe_class_name)
        
        dict_info["sample_count"] = stats["migrated_files"]
        dict_info["updated_at"] = datetime.now().isoformat()
        
        # Save the updated metadata
        service._save_metadata()
        logging.info("Updated dictionary metadata")
    except Exception as e:
        logging.error(f"Failed to update metadata: {e}")
    
    # Print summary
    logging.info("\nMigration Summary:")
    logging.info(f"Total files processed: {stats['total_files']}")
    logging.info(f"Files migrated: {stats['migrated_files']}")
    logging.info(f"Classes created: {stats['classes_created']}")
    logging.info(f"Classes: {', '.join(stats['classes'])}")
    
    return stats

if __name__ == "__main__":
    logging.info("Starting sound file migration...")
    stats = migrate_sounds()
    logging.info("Migration complete!") 