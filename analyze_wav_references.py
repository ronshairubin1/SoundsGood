#!/usr/bin/env python3
"""
Script to analyze all references to .wav files in the codebase.

This script:
1. Searches for all instances of ".wav" in the codebase
2. Analyzes each occurrence to determine what operation is being performed
3. Exports the results to a markdown file
"""

import os
import re
import subprocess
import sys
from datetime import datetime

# Configure paths to exclude
EXCLUDED_PATHS = [
    'legacy/',
    'get_code/',
    'data/sounds/',
    '.git/',
    '__pycache__/',
    'venv/',
    '.vscode/',
    'analyze_wav_references.py',  # Exclude this script itself
]

# Output file
OUTPUT_FILE = 'wav_file_references.md'

def should_exclude(path):
    """Check if a path should be excluded from analysis."""
    for excluded in EXCLUDED_PATHS:
        if excluded in path:
            return True
    return False

def find_wav_references():
    """Find all references to .wav files in the codebase."""
    # Use grep to find all instances of .wav in the codebase
    cmd = ['grep', '-r', '--include=*.py', '--include=*.html', '--include=*.js', '--include=*.md', 
           '--exclude-dir=legacy', '--exclude-dir=get_code', '--exclude-dir=data/sounds', 
           '--exclude-dir=.git', '--exclude-dir=__pycache__', '--exclude-dir=venv', '.wav', '.']
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        lines = result.stdout.splitlines()
        
        # Additional filtering if needed
        filtered_lines = [line for line in lines if not should_exclude(line)]
        
        return filtered_lines
    except Exception as e:
        print(f"Error running grep: {e}")
        return []

def analyze_reference(line):
    """Analyze a line containing a reference to a .wav file."""
    # Extract file path and line number
    parts = line.split(':', 1)
    if len(parts) < 2:
        return None
    
    file_path = parts[0]
    if should_exclude(file_path):
        return None
    
    content = parts[1]
    
    # Determine the operation being performed
    operation = "Unknown"
    location = "Unknown"
    
    # Check for common patterns
    if "open(" in content and ".wav" in content:
        operation = "Reading file"
        # Try to extract the file path
        matches = re.findall(r'open\([\'"](.+?\.wav)[\'"]', content)
        if matches:
            location = matches[0]
        else:
            location = "Dynamic path"
    elif "save(" in content and ".wav" in content:
        operation = "Writing file"
        matches = re.findall(r'save\([\'"](.+?\.wav)[\'"]', content)
        if matches:
            location = matches[0]
        else:
            location = "Dynamic path"
    elif "export(" in content and ".wav" in content:
        operation = "Exporting file"
        matches = re.findall(r'export\([\'"](.+?\.wav)[\'"]', content)
        if matches:
            location = matches[0]
        else:
            location = "Dynamic path"
    elif "os.path.join" in content and ".wav" in content:
        operation = "Path construction"
        # Try to find the base directory
        base_dir_match = re.search(r'os\.path\.join\(([^,]+)', content)
        if base_dir_match:
            base_dir = base_dir_match.group(1).strip()
            location = f"Based on {base_dir}"
        else:
            location = "Dynamic path"
    elif "endswith" in content and ".wav" in content:
        operation = "File filtering"
        location = "Checking file extension"
    elif "content_type" in content and ".wav" in content:
        operation = "MIME type handling"
        location = "HTTP response"
    elif "url_for" in content and ".wav" in content:
        operation = "URL generation"
        location = "Web routing"
    elif "src=" in content and ".wav" in content:
        operation = "HTML audio element"
        matches = re.findall(r'src=[\'"](.+?\.wav)[\'"]', content)
        if matches:
            location = matches[0]
        else:
            location = "Dynamic path"
    elif "send_from_directory" in content and ".wav" in content:
        operation = "Serving file"
        dir_match = re.search(r'send_from_directory\(([^,]+)', content)
        if dir_match:
            directory = dir_match.group(1).strip()
            location = f"From {directory}"
        else:
            location = "Dynamic directory"
    elif "path:" in content and ".wav" in content:
        operation = "Path definition"
        matches = re.findall(r'path:[\'"](.+?\.wav)[\'"]', content)
        if matches:
            location = matches[0]
        else:
            location = "Path template"
    elif "import" in content and ".wav" in content:
        operation = "Audio processing library"
        location = "Library import"
    
    # If we couldn't determine the operation, provide the line for manual analysis
    if operation == "Unknown":
        operation = "See line content"
        location = content.strip()
    
    return {
        'file_path': file_path,
        'content': content.strip(),
        'operation': operation,
        'location': location
    }

def generate_markdown(references):
    """Generate a markdown file with the analysis results."""
    with open(OUTPUT_FILE, 'w') as f:
        f.write("# Analysis of .wav File References in Codebase\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("This document contains an analysis of all references to `.wav` files in the codebase, excluding legacy code and sound data files.\n\n")
        
        f.write("## Table of Contents\n\n")
        f.write("1. [Summary](#summary)\n")
        f.write("2. [References by File](#references-by-file)\n")
        f.write("3. [References by Operation](#references-by-operation)\n\n")
        
        # Generate summary
        f.write("## Summary\n\n")
        f.write(f"Total .wav file references: {len(references)}\n\n")
        
        # Count operations
        operations = {}
        for ref in references:
            op = ref['operation']
            if op in operations:
                operations[op] += 1
            else:
                operations[op] = 1
        
        f.write("Operation types:\n\n")
        for op, count in sorted(operations.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- {op}: {count}\n")
        f.write("\n")
        
        # Group references by file
        files = {}
        for ref in references:
            file_path = ref['file_path']
            if file_path in files:
                files[file_path].append(ref)
            else:
                files[file_path] = [ref]
        
        f.write("## References by File\n\n")
        for file_path, refs in sorted(files.items()):
            f.write(f"### {file_path}\n\n")
            f.write("| Line | Operation | Location | Content |\n")
            f.write("|------|-----------|----------|--------|\n")
            
            for ref in refs:
                f.write(f"| {ref['file_path']} | {ref['operation']} | {ref['location']} | `{ref['content']}` |\n")
            
            f.write("\n")
        
        # Group references by operation
        by_operation = {}
        for ref in references:
            op = ref['operation']
            if op in by_operation:
                by_operation[op].append(ref)
            else:
                by_operation[op] = [ref]
        
        f.write("## References by Operation\n\n")
        for op, refs in sorted(by_operation.items()):
            f.write(f"### {op}\n\n")
            f.write("| File | Location | Content |\n")
            f.write("|------|----------|--------|\n")
            
            for ref in refs:
                f.write(f"| {ref['file_path']} | {ref['location']} | `{ref['content']}` |\n")
            
            f.write("\n")

def main():
    """Main function to find and analyze .wav file references."""
    print("Finding .wav file references in the codebase...")
    references = find_wav_references()
    
    print(f"Found {len(references)} references to .wav files.")
    
    print("Analyzing references...")
    analyzed_refs = []
    for line in references:
        ref = analyze_reference(line)
        if ref:
            analyzed_refs.append(ref)
    
    print(f"Analyzed {len(analyzed_refs)} references.")
    
    print(f"Generating markdown file: {OUTPUT_FILE}")
    generate_markdown(analyzed_refs)
    
    print("Done!")

if __name__ == "__main__":
    main() 