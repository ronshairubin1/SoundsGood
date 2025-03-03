#!/usr/bin/env python3
"""
Script to identify all folders where .wav files are read from or written to in the codebase.

This script uses manual path mappings based on code analysis to ensure accuracy.
"""

import re
import subprocess

# Output file
OUTPUT_FILE = 'wav_file_destinations.txt'

# Known paths used in the application
WAV_FILE_PATHS = {
    # Main sound directories
    "data/sounds/training_sounds/": {
        "purpose": "Main sound storage directory containing verified sound files used for training models",
        "operations": ["READ", "WRITE"],
        "referenced_as": ["Config.TRAINING_SOUNDS_DIR", "Config.SOUNDS_DIR", "class_path"]
    },
    "data/sounds/training_sounds/{class}/": {
        "purpose": "Class-specific directories containing sound samples for each class (e.g., 'ah', 'eh', etc.)",
        "operations": ["READ", "WRITE"],
        "referenced_as": ["os.path.join(Config.TRAINING_SOUNDS_DIR, class_name)"]
    },
    "data/sounds/raw_sounds/": {
        "purpose": "Storage for original unprocessed recordings before chunking",
        "operations": ["WRITE"],
        "referenced_as": ["Config.RAW_SOUNDS_DIR", "raw_sounds_dir"]
    },
    "data/sounds/raw_sounds/{class}/": {
        "purpose": "Class-specific directories for raw recordings",
        "operations": ["WRITE"],
        "referenced_as": ["os.path.join(Config.RAW_SOUNDS_DIR, class_name)"]
    },
    "data/sounds/temp_sounds/": {
        "purpose": "Temporary storage for sound chunks during processing",
        "operations": ["WRITE"],
        "referenced_as": ["Config.TEMP_SOUNDS_DIR", "temp_sounds_dir"]
    },
    "data/sounds/temp_sounds/{class}/": {
        "purpose": "Class-specific directories for temporary sound chunks",
        "operations": ["WRITE"],
        "referenced_as": ["os.path.join(Config.TEMP_SOUNDS_DIR, class_name)"]
    },
    "root/temp/": {
        "purpose": "Temporary files for ML operations (predictions, processing)",
        "operations": ["WRITE", "READ"],
        "referenced_as": ["Config.TEMP_DIR", "temp_path"]
    },
    "root/": {
        "purpose": "Root directory - used mainly for test recordings",
        "operations": ["WRITE"],
        "referenced_as": ["test_recording.wav"]
    }
}

def find_wav_references():
    """Find all references to .wav files in the codebase, excluding legacy and other specific directories."""
    cmd = [
        'grep', '-r', 
        '--include=*.py', '--include=*.html', '--include=*.js',
        '--exclude-dir=legacy', '--exclude-dir=get_code', '--exclude-dir=.git',
        '--exclude-dir=__pycache__', '--exclude-dir=venv',
        '.wav', '.'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.splitlines()

def get_directory_references():
    """Find references to directories that might be used for .wav files."""
    directory_patterns = [
        r'Config\.SOUNDS_DIR', 
        r'Config\.TRAINING_SOUNDS_DIR',
        r'Config\.RAW_SOUNDS_DIR', 
        r'Config\.TEMP_SOUNDS_DIR',
        r'Config\.TEMP_DIR',
        r'class_path',
        r'raw_sounds_dir',
        r'temp_sounds_dir'
    ]
    
    pattern = '|'.join(directory_patterns)
    cmd = ['grep', '-r', '--include=*.py', pattern, '.']
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.splitlines()

def analyze_sound_directory_usage():
    """Count references to different sound directories in the codebase."""
    wav_references = find_wav_references()
    directory_references = get_directory_references()
    
    # Count actual references in code
    reference_counts = {path: 0 for path in WAV_FILE_PATHS}
    
    # Check wav references
    for line in wav_references:
        for path, info in WAV_FILE_PATHS.items():
            for ref in info["referenced_as"]:
                if ref in line:
                    reference_counts[path] += 1
    
    # Check directory references
    for line in directory_references:
        for path, info in WAV_FILE_PATHS.items():
            for ref in info["referenced_as"]:
                if ref in line and ".wav" in line:
                    reference_counts[path] += 1
    
    return reference_counts

def generate_output_file(reference_counts):
    """Generate the output file with wav file locations analysis."""
    with open(OUTPUT_FILE, 'w') as f:
        f.write("# .wav File Locations in Codebase\n\n")
        f.write("This file shows where .wav files are read from and written to in the codebase.\n\n")
        
        # Writing the path information
        f.write("## Sound File Directories\n\n")
        f.write("| Directory | Purpose | Operations | References |\n")
        f.write("|-----------|---------|------------|------------|\n")
        
        for path, info in sorted(WAV_FILE_PATHS.items()):
            operations = ", ".join(info["operations"])
            operation_symbols = ""
            if "READ" in info["operations"] and "WRITE" in info["operations"]:
                operation_symbols = "<-->"
            elif "READ" in info["operations"]:
                operation_symbols = "<--"
            elif "WRITE" in info["operations"]:
                operation_symbols = "-->"
            
            f.write(f"| {path} | {info['purpose']} | {operations} {operation_symbols} | {reference_counts[path]} |\n")
        
        # Reading and writing explained
        f.write("\n## File Operations Explained\n\n")
        
        f.write("### Reading Operations (Files Sourced From)\n\n")
        for path, info in sorted(WAV_FILE_PATHS.items()):
            if "READ" in info["operations"]:
                # Calculate operation symbol for this specific path
                read_op_symbol = "<--" if "WRITE" not in info["operations"] else "<-->"
                f.write(f"- **{path}** {read_op_symbol} .wav\n")
                f.write(f"  - Purpose: {info['purpose']}\n")
                f.write(f"  - Referenced as: {', '.join(info['referenced_as'])}\n\n")
        
        f.write("### Writing Operations (Files Written To)\n\n")
        for path, info in sorted(WAV_FILE_PATHS.items()):
            if "WRITE" in info["operations"]:
                # Calculate operation symbol for this specific path
                write_op_symbol = "-->" if "READ" not in info["operations"] else "<-->"
                f.write(f"- **{path}** {write_op_symbol} .wav\n")
                f.write(f"  - Purpose: {info['purpose']}\n")
                f.write(f"  - Referenced as: {', '.join(info['referenced_as'])}\n\n")
        
        # Simple full list for quick reference
        f.write("\n## Simple List of All Sound File Paths\n\n")
        for path, info in sorted(WAV_FILE_PATHS.items()):
            operations = ", ".join(info["operations"])
            
            if "READ" in info["operations"] and "WRITE" in info["operations"]:
                operation_symbols = "<-->"
            elif "READ" in info["operations"]:
                operation_symbols = "<--"
            elif "WRITE" in info["operations"]:
                operation_symbols = "-->"
                
            f.write(f"{path} {operation_symbols} .wav\n")

def main():
    """Main function to analyze and report on .wav file locations."""
    print("Analyzing .wav file directory usage in the codebase...")
    reference_counts = analyze_sound_directory_usage()
    
    print("Generating output file...")
    generate_output_file(reference_counts)
    
    print(f"Analysis complete! Results written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main() 