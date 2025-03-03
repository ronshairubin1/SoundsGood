#!/usr/bin/env python3
"""
Script to identify all folders where .wav files are read from or written to in the codebase.

Output format:
folder/path/ <-- .wav (when files are read from this location)
folder/path/ --> .wav (when files are written to this location)
"""

import os
import re
import subprocess
import sys
from collections import defaultdict

# Configure paths to exclude
EXCLUDED_PATHS = [
    'legacy/',
    'get_code/',
    'data/sounds/',
    '.git/',
    '__pycache__/',
    'venv/',
    '.vscode/',
    'analyze_wav_references.py',
    'analyze_wav_paths.py',
    'track_wav_folders.py',
    '*.md',  # Exclude markdown files
]

# Output file
OUTPUT_FILE = 'wav_file_locations.txt'

def find_wav_references():
    """Find all references to .wav files in the codebase."""
    # Use grep to find all instances of .wav in Python, HTML, and JS files
    cmd = [
        'grep', '-r', 
        '--include=*.py', '--include=*.html', '--include=*.js',
        '--exclude-dir=legacy', '--exclude-dir=get_code', '--exclude-dir=data/sounds',
        '--exclude-dir=.git', '--exclude-dir=__pycache__', '--exclude-dir=venv',
        '.wav', '.'
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        lines = result.stdout.splitlines()
        
        # Filter out any excluded paths
        filtered_lines = []
        for line in lines:
            excluded = False
            for excluded_path in EXCLUDED_PATHS:
                if excluded_path.startswith('*'):
                    if excluded_path[1:] in line:
                        excluded = True
                        break
                elif excluded_path in line:
                    excluded = True
                    break
            
            if not excluded:
                filtered_lines.append(line)
        
        return filtered_lines
    except Exception as e:
        print(f"Error running grep: {e}")
        return []

def determine_operation_type(content):
    """Determine if the operation is reading or writing a .wav file."""
    # Patterns that indicate reading operations
    read_patterns = [
        r'open\(\s*[\'"][^\'"]*.wav[\'"]',  # open("file.wav")
        r'load\(\s*[\'"][^\'"]*.wav[\'"]',  # load("file.wav")
        r'wavfile\.read\([\'"][^\'"]*.wav[\'"]',  # wavfile.read("file.wav")
        r'wave\.open\([\'"][^\'"]*.wav[\'"]',  # wave.open("file.wav")
        r'from_file\([\'"][^\'"]*.wav[\'"]',  # from_file("file.wav")
        r'wavesurfer\.load\(',  # wavesurfer.load()
    ]
    
    # Patterns that indicate writing operations
    write_patterns = [
        r'wavfile\.write\([\'"][^\'"]*.wav[\'"]',  # wavfile.write("file.wav")
        r'export\([\'"][^\'"]*.wav[\'"]',  # export("file.wav", format="wav")
        r'save\([\'"][^\'"]*.wav[\'"]',  # save("file.wav")
        r'audio\.export\([^,]+,\s*format=[\'"]wav[\'"]',  # audio.export(path, format="wav")
        r'formData\.append\(.*\.wav',  # formData.append(file, "recording.wav")
    ]
    
    # Check for read patterns
    for pattern in read_patterns:
        if re.search(pattern, content):
            return "READ"
    
    # Check for write patterns
    for pattern in write_patterns:
        if re.search(pattern, content):
            return "WRITE"
    
    # If it contains a path construction but not clearly read/write, check context
    if "os.path.join" in content:
        if any(word in content for word in ["save", "write", "export", "create", "open(", "new"]):
            return "WRITE"
        if any(word in content for word in ["load", "read", "get", "fetch"]):
            return "READ"
    
    # If none of the above, use heuristics based on variable names and context
    if any(word in content for word in ["save", "write", "export", "create", "record"]):
        return "WRITE"
    if any(word in content for word in ["load", "read", "get", "fetch", "open"]):
        return "READ"
    
    # If it's an endswith check, it's likely reading
    if "endswith('.wav')" in content:
        return "READ"
    
    # Default to unknown
    return "UNKNOWN"

def extract_folder_from_path(path_str, file_content):
    """Extract the folder path from a file path string or variable."""
    # Handle different path construction patterns
    
    # Check for direct Config variables in the line
    config_patterns = [
        (r'Config\.TEMP_DIR', "root/temp/"),
        (r'Config\.SOUNDS_DIR', "data/sounds/training_sounds/"),  # Updated to consolidated location
        (r'Config\.TRAINING_SOUNDS_DIR', "data/sounds/training_sounds/"),
        (r'Config\.RAW_SOUNDS_DIR', "data/sounds/raw_sounds/"),
        (r'raw_sounds_dir', "data/sounds/raw_sounds/"),
        (r'temp_sounds_dir', "data/sounds/temp_sounds/")
    ]
    
    for pattern, folder in config_patterns:
        if re.search(pattern, path_str) or re.search(pattern, file_content):
            return folder
    
    # Check for direct path construction with os.path.join
    if "os.path.join" in path_str:
        # Try to extract the directory parts of the path
        dir_match = re.search(r'os\.path\.join\(([^,]+)', path_str)
        if dir_match:
            dir_var = dir_match.group(1).strip()
            
            # Map common directory variables to paths
            config_dirs = {
                "Config.TEMP_DIR": "root/temp/",
                "Config.SOUNDS_DIR": "data/sounds/training_sounds/",
                "Config.TRAINING_SOUNDS_DIR": "data/sounds/training_sounds/",
                "Config.RAW_SOUNDS_DIR": "data/sounds/raw_sounds/",
                "raw_sounds_dir": "data/sounds/raw_sounds/",
                "class_path": "data/sounds/training_sounds/{class}/",
                "temp_sounds_dir": "data/sounds/temp_sounds/"
            }
            
            for var_name, folder_path in config_dirs.items():
                if var_name in dir_var:
                    return folder_path
            
            # Check for specific directory names
            if "class_path" in path_str or "class_dir" in path_str:
                return "data/sounds/training_sounds/{class}/"
            
            # Check if the variable likely refers to a Config path
            if "Config." in dir_var:
                # Try to determine which Config directory it is
                if "TEMP" in dir_var:
                    return "root/temp/"
                if "RAW" in dir_var:
                    return "data/sounds/raw_sounds/"
                if "TRAINING" in dir_var:
                    return "data/sounds/training_sounds/"
                if "SOUNDS" in dir_var:
                    return "data/sounds/training_sounds/"  # Updated to consolidated location
                
                return f"[Config path: {dir_var}]"
            
            return f"[Variable: {dir_var}]"
    
    # Check for explicit class_ references    
    if "class_path" in path_str or "class_dir" in path_str or "class_" in path_str:
        return "data/sounds/training_sounds/{class}/"
        
    # Direct path patterns for .wav files
    path_match = re.search(r'[\'"]([^\'\"]*/)([^\/\'\"]+\.wav)[\'"]', path_str)
    if path_match:
        folder = path_match.group(1)
        if folder:
            return folder
    
    # Handle special cases
    if "test_recording.wav" in path_str:
        return "root/"
    
    # Look for common file operations
    if "endswith('.wav')" in path_str:
        if "files = [f for f in os.listdir" in path_str or "samples = [f for f in os.listdir" in path_str:
            # Try to get the directory from listdir patterns
            listdir_match = re.search(r'listdir\(([^)]+)\)', path_str)
            if listdir_match:
                listdir_arg = listdir_match.group(1).strip()
                if "class_path" in listdir_arg or "class_dir" in listdir_arg:
                    return "data/sounds/training_sounds/{class}/"
                if "sound_dir" in listdir_arg:
                    return "data/sounds/training_sounds/{class}/"
                if "Config" in listdir_arg:
                    if "TRAINING" in listdir_arg:
                        return "data/sounds/training_sounds/"
                    if "SOUNDS" in listdir_arg:
                        return "data/sounds/training_sounds/"  # Updated to consolidated location
                    if "RAW" in listdir_arg:
                        return "data/sounds/raw_sounds/"
    
    # Look for sound class handling
    if "sound_class" in path_str or "class_name" in path_str:
        if "raw_" in path_str or "RAW_" in path_str:
            return "data/sounds/raw_sounds/{class}/"
        return "data/sounds/training_sounds/{class}/"
    
    # Look for common patterns in the content
    if "TEMP_DIR" in file_content:
        return "root/temp/"
    if "RAW_SOUNDS_DIR" in file_content:
        return "data/sounds/raw_sounds/"
    if "TRAINING_SOUNDS_DIR" in file_content:
        return "data/sounds/training_sounds/"
    if "SOUNDS_DIR" in file_content:
        return "data/sounds/training_sounds/"  # Updated to consolidated location
    
    # Default if we can't determine
    return "[Unknown folder]"

def analyze_references(references):
    """Analyze references to extract folder locations."""
    folder_operations = defaultdict(set)  # {folder: set(operations)}
    folder_sources = defaultdict(list)    # {folder: [source_files]}
    
    for line in references:
        parts = line.split(':', 1)
        if len(parts) < 2:
            continue
            
        file_path = parts[0]
        content = parts[1].strip()
        
        # Skip if this isn't actually about .wav files
        if not re.search(r'\.wav', content):
            continue
        
        # Skip UI and icon references
        if ('soundwave' in content and '.wav' not in content) or \
           ('waveform' in content and '.wav' not in content) or \
           ('WaveSurfer' in content and '.wav' not in content):
            continue
        
        # Determine operation type (READ or WRITE)
        operation_type = determine_operation_type(content)
        
        # Try to extract the folder path
        folder = extract_folder_from_path(content, content)
        
        # Record the operation even if UNKNOWN, as we still want to see the folder
        folder_operations[folder].add(operation_type)
        folder_sources[folder].append((file_path, content))
    
    return folder_operations, folder_sources

def generate_output(folder_operations, folder_sources):
    """Generate output file with folder paths and operations."""
    with open(OUTPUT_FILE, 'w') as f:
        f.write("# .wav File Locations in Codebase\n\n")
        f.write("This file shows where .wav files are read from and written to in the codebase.\n\n")
        f.write("- `<--` indicates files are read from this location\n")
        f.write("- `-->` indicates files are written to this location\n\n")
        
        # First list folders with both operations
        f.write("## Folders with both read and write operations\n\n")
        for folder, operations in sorted(folder_operations.items()):
            if "READ" in operations and "WRITE" in operations:
                f.write(f"{folder} <--> .wav\n")
                # Add example references
                for source_file, content in folder_sources[folder][:2]:  # Limit to 2 examples per folder
                    f.write(f"  - {source_file}: {content[:100]}...\n" if len(content) > 100 else f"  - {source_file}: {content}\n")
                f.write("\n")
        
        # Then list read-only folders
        f.write("\n## Folders where .wav files are read from\n\n")
        for folder, operations in sorted(folder_operations.items()):
            if "READ" in operations and "WRITE" not in operations:
                f.write(f"{folder} <-- .wav\n")
                # Add example references
                for source_file, content in folder_sources[folder][:2]:  # Limit to 2 examples per folder
                    f.write(f"  - {source_file}: {content[:100]}...\n" if len(content) > 100 else f"  - {source_file}: {content}\n")
                f.write("\n")
        
        # List write-only folders
        f.write("\n## Folders where .wav files are written to\n\n")
        for folder, operations in sorted(folder_operations.items()):
            if "WRITE" in operations and "READ" not in operations:
                f.write(f"{folder} --> .wav\n")
                # Add example references
                for source_file, content in folder_sources[folder][:2]:  # Limit to 2 examples per folder
                    f.write(f"  - {source_file}: {content[:100]}...\n" if len(content) > 100 else f"  - {source_file}: {content}\n")
                f.write("\n")
        
        # List folders with unknown operations
        unknown_folders = [folder for folder, ops in folder_operations.items() 
                         if "UNKNOWN" in ops and "READ" not in ops and "WRITE" not in ops]
        
        if unknown_folders:
            f.write("\n## Folders with unknown operations (might be either read or write)\n\n")
            for folder in sorted(unknown_folders):
                f.write(f"{folder} .wav (unknown operation)\n")
                # Add example references
                for source_file, content in folder_sources[folder][:2]:  # Limit to 2 examples per folder
                    f.write(f"  - {source_file}: {content[:100]}...\n" if len(content) > 100 else f"  - {source_file}: {content}\n")
                f.write("\n")
                
        # List all unique folders identified in a simple format for quick reference
        f.write("\n## Simple List of All Identified Folders\n\n")
        for folder in sorted(folder_operations.keys()):
            operations = folder_operations[folder]
            direction = ""
            if "READ" in operations and "WRITE" in operations:
                direction = "<-->"
            elif "READ" in operations:
                direction = "<--"
            elif "WRITE" in operations:
                direction = "-->"
            else:
                direction = "???"
                
            f.write(f"{folder} {direction} .wav\n")
        
        # Add summary
        f.write("\n## Summary\n\n")
        read_write = sum(1 for ops in folder_operations.values() if "READ" in ops and "WRITE" in ops)
        read_only = sum(1 for ops in folder_operations.values() if "READ" in ops and "WRITE" not in ops)
        write_only = sum(1 for ops in folder_operations.values() if "WRITE" in ops and "READ" not in ops)
        unknown = sum(1 for ops in folder_operations.values() 
                     if "UNKNOWN" in ops and "READ" not in ops and "WRITE" not in ops)
        
        f.write(f"- Total folders: {len(folder_operations)}\n")
        f.write(f"- Folders with both read and write operations: {read_write}\n")
        f.write(f"- Folders with read-only operations: {read_only}\n")
        f.write(f"- Folders with write-only operations: {write_only}\n")
        f.write(f"- Folders with unknown operations: {unknown}\n")

def main():
    """Main function to find and analyze .wav file locations."""
    print("Finding .wav file references in the codebase...")
    references = find_wav_references()
    
    print(f"Found {len(references)} references to .wav files.")
    
    print("Analyzing folder locations...")
    folder_operations, folder_sources = analyze_references(references)
    
    print(f"Identified {len(folder_operations)} distinct folder locations.")
    
    print(f"Generating output file: {OUTPUT_FILE}")
    generate_output(folder_operations, folder_sources)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main() 