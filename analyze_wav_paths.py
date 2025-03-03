#!/usr/bin/env python3
"""
Script to analyze how .wav files are handled in the codebase.

This script:
1. Identifies all paths where .wav files are read from or written to
2. Categorizes operations (read, write, path construction, etc.)
3. Outputs a summary of findings directly to the console
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
    '*.md',  # Exclude markdown files
]

# Categories for .wav file operations
CATEGORIES = {
    'READ': ['open(', 'read(', 'load(', 'wavfile.read', 'wave.open', 'from_file'],
    'WRITE': ['write(', 'wavfile.write', 'save(', 'export(', 'AudioSegment', 'wf.writeframes'],
    'PATH_CONSTRUCTION': ['os.path.join', 'path =', 'filepath =', 'file_path =', '= os.path'],
    'SERVE': ['send_from_directory', 'send_file', 'url_for', 'src='],
    'FILE_CHECK': ['endswith', 'if f.', 'if filename.'],
}

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

def categorize_reference(line):
    """Categorize a line containing a reference to a .wav file."""
    # Extract file path and content
    parts = line.split(':', 1)
    if len(parts) < 2:
        return None
    
    file_path = parts[0]
    content = parts[1].strip()
    
    # Skip icon and waveform references that aren't related to actual .wav files
    if 'soundwave' in content and 'wav' not in content:
        return None
    if 'waveform' in content and '.wav' not in content:
        return None
    if 'WaveSurfer' in content and '.wav' not in content:
        return None
    
    # Determine the operation category
    category = "OTHER"
    for cat, patterns in CATEGORIES.items():
        for pattern in patterns:
            if pattern in content:
                category = cat
                break
        if category != "OTHER":
            break
    
    # Try to extract relevant file paths or variables
    file_path_match = None
    
    # Look for direct file paths
    if '.wav' in content:
        # Check for complete path patterns
        path_patterns = [
            r'(?:[\'\"])((?:/|\./)?\w+(?:/\w+)*\.wav)(?:[\'\"])',  # Quoted path
            r'os\.path\.join\([^,]+,\s*[^,]+,\s*[\'\"]([^\'\"]+\.wav)[\'\"]',  # os.path.join with wav filename
            r'filename\s*=\s*[\'\"]([^\'\"]+\.wav)[\'\"]',  # filename assignment
            r'filepath\s*=\s*[\'\"]([^\'\"]+\.wav)[\'\"]',  # filepath assignment
        ]
        
        for pattern in path_patterns:
            matches = re.findall(pattern, content)
            if matches:
                file_path_match = matches[0]
                break
    
    # Extract variables used for path construction
    path_vars = []
    if category == 'PATH_CONSTRUCTION':
        # Look for directory variables
        dir_vars = re.findall(r'os\.path\.join\((\w+)', content)
        if dir_vars:
            path_vars.append(dir_vars[0])
        
        # Look for filename variables
        filename_vars = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\.wav', content)
        if filename_vars:
            path_vars.append(f"{filename_vars[0]}.wav")
        
        # Look for f-strings with .wav
        fstring_vars = re.findall(r'f[\'\"](.+?\.wav)[\'\"]', content)
        if fstring_vars:
            path_vars.append(f"f-string: {fstring_vars[0]}")
    
    return {
        'file': file_path,
        'category': category,
        'content': content,
        'path_match': file_path_match,
        'path_vars': path_vars,
    }

def analyze_references(references):
    """Analyze and group references by category and file."""
    categories = defaultdict(list)
    files = defaultdict(list)
    
    for ref in references:
        if ref:
            categories[ref['category']].append(ref)
            files[ref['file']].append(ref)
    
    return categories, files

def print_summary(categories, files):
    """Print a summary of the findings."""
    print("\n===== SUMMARY OF .WAV FILE USAGE IN CODEBASE =====\n")
    
    # Print category counts
    print("CATEGORIES OF .WAV FILE OPERATIONS:")
    for category, refs in categories.items():
        print(f"  {category}: {len(refs)} references")
    print()
    
    # Print detailed category information
    for category, refs in categories.items():
        print(f"=== {category} OPERATIONS ===")
        for ref in refs:
            file_info = f"{ref['file']}"
            path_info = ""
            if ref['path_match']:
                path_info = f"PATH: {ref['path_match']}"
            elif ref['path_vars']:
                path_info = f"VARS: {', '.join(ref['path_vars'])}"
            
            print(f"  {file_info}")
            if path_info:
                print(f"    {path_info}")
            print(f"    {ref['content'][:100]}..." if len(ref['content']) > 100 else f"    {ref['content']}")
            print()
        print()
    
    # Print files with the most .wav references
    print("TOP FILES WITH .WAV REFERENCES:")
    top_files = sorted(files.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    for file_path, refs in top_files:
        print(f"  {file_path}: {len(refs)} references")
    print()

def main():
    """Main function to find and analyze .wav file references."""
    print("Finding .wav file references in the codebase...")
    references = find_wav_references()
    
    print(f"Found {len(references)} references to .wav files.")
    
    print("Analyzing and categorizing references...")
    
    categorized_refs = []
    for line in references:
        ref = categorize_reference(line)
        if ref:
            categorized_refs.append(ref)
    
    categories, files = analyze_references(categorized_refs)
    
    print(f"Analyzed {len(categorized_refs)} references.")
    
    print_summary(categories, files)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main() 