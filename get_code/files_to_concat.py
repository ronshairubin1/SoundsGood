import os
import sys
from pathlib import Path
from typing import List, Dict, Set

def find_files(
    start_path: str,
    ignore_patterns: Set[str] = None,
    ignore_extensions: Set[str] = None,
    sample_extensions: Dict[str, int] = None,
    ignore_filenames: Set[str] = None
) -> List[str]:
    """
    Find files recursively from start_path, with filtering options.
    
    Args:
        start_path: Directory to start searching from.
        ignore_patterns: Set of patterns to ignore in file/folder names.
        ignore_extensions: Set of file extensions to ignore.
        sample_extensions: Dict of extensions and how many examples to keep,
                           e.g., {'.log': 1} to keep only the largest log file.
        ignore_filenames: Set of exact filenames to ignore.
    
    Returns:
        List of relative file paths (relative to start_path).
    """
    if ignore_patterns is None:
        ignore_patterns = {
            '.git', '__pycache__', 'node_modules', '.env', '.DS_Store',
            'get_code',
            '__init__.py',
            'yarn.lock',
            'package-lock.json'
        }
    
    if ignore_extensions is None:
        ignore_extensions = {'.pyc', '.pyo', '.pyd', '.so', '.dll', '.dylib', '.wav'}
    
    if sample_extensions is None:
        sample_extensions = {}
    
    if ignore_filenames is None:
        ignore_filenames = {'__init__.py', 'yarn.lock', 'package-lock.json'}
    
    # Dictionary to store files by extension for sampling
    extension_files = {ext: [] for ext in sample_extensions}
    
    # List to store all found files
    found_files = []
    
    # Convert start_path to absolute path
    start_path = os.path.abspath(start_path)
    
    for root, dirs, files in os.walk(start_path):
        # Remove directories that match ignore patterns exactly or are direct children
        dirs[:] = [d for d in dirs 
                  if d not in ignore_patterns and 
                  not any(root.endswith(os.path.sep + pat) for pat in ignore_patterns)]
        
        for file in files:
            # Skip exact filename matches
            if file in ignore_filenames:
                continue
                
            # Skip files with ignored extensions
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext in ignore_extensions:
                continue
            
            # Skip files in ignored paths
            rel_path = os.path.relpath(os.path.join(root, file), start_path)
            path_parts = rel_path.split(os.path.sep)
            if any(part in ignore_patterns for part in path_parts):
                continue
                
            found_files.append(rel_path)
    
    # Process sampled extensions: for example, keeping only the largest file for a given extension
    for ext, files in extension_files.items():
        if not files:
            continue
        
        # Sort by file size (largest first)
        sorted_files = sorted(files, key=lambda x: os.path.getsize(x[0]), reverse=True)
        
        # Take the specified number of samples
        samples = sorted_files[:sample_extensions[ext]]
        found_files.extend(sample[1] for sample in samples)
    
    return sorted(found_files)

def create_concat_list(
    output_file: str = "files_to_concat.txt",
    start_path: str = None
) -> None:
    """
    Create a list of files to concatenate.
    
    Special rules applied:
      - Files with "resources" in their path and any files in the get_code folder are excluded.
      - From backend/conversations, only the first two and the last (alphabetically) are included.
      - From backend/logs/conversations, only files matching the same base names in the sample
        (i.e. before the extension) are included.
    
    The start_path is set to the repository root (the parent of the get_code folder) so that
    relative paths are computed from the repository root. The output file is written to the
    get_code folder.
    """
    # Determine the repository root.
    # Assumes this script is in get_code (a subfolder of the repository root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    
    # Define ignore patterns and extensions
    ignore_patterns = {
        '.git', '__pycache__', 'node_modules', '.env', '.DS_Store',
        'venv', '.idea', '.vscode', 'build', 'dist'
    }
    ignore_extensions = {
        '.pyc', '.pyo', '.pyd', '.so', '.dll', '.dylib',
        '.exe', '.bin', '.obj', '.cache','.log','.wav'
    }
    ignore_filenames = {
        '__init__.py', 'yarn.lock', 'package-lock.json',
        'README.md', 'Dockerfile',
        '.python-version',
        'README.txt',
        '*.wav'
    }
    
    # No generic sampling here—we handle special sampling for conversations separately.
    sample_extensions = {}
    
    try:
        # Collect all files (relative to the repository root)
        all_files = find_files(
            repo_root,
            ignore_patterns=ignore_patterns,
            ignore_extensions=ignore_extensions,
            sample_extensions=sample_extensions,
            ignore_filenames=ignore_filenames
        )
        
        # Exclude files that are in any "resources" folder or in the "get_code" folder.
        filtered_files = []
        for f in all_files:
            f_lower = f.lower()
            # Exclude if "resources" appears anywhere in the path.
            if "resources" in f_lower:
                continue
            # Exclude if the file is in the get_code folder.
            if f.startswith("get_code" + os.sep) or f.endswith('.json') or f.endswith('.h5') or f.endswith('.joblib') or f.endswith('.gitignore') or f.endswith('.pickle') or f.endswith('.pkl') or f.endswith('.pth') or f.endswith('.pt') or f.endswith('.csv') or f.endswith('.txt') or f.endswith('.md') or f.endswith('.yaml') or f.endswith('.yml') or f.endswith('.jsonl') or f.endswith('.jsonl.gz') or f.endswith('.jsonl.bz2') or f.endswith('.jsonl.zip') or f.endswith('.jsonl.tar') or f.endswith('.jsonl.tar.gz') or f.endswith('.jsonl.tar.bz2') or f.endswith('.jsonl.tar.zip') or f.endswith('.sh') or f.endswith('.code-workspace') or f.endswith('.cursorignore'):
                continue
            filtered_files.append(f)
        
        # Define prefixes for the special directories.
        conv_prefix = os.path.join("backend", "conversations") + os.sep
        log_conv_prefix = os.path.join("backend", "logs", "conversations") + os.sep
        
        # Extract files in the backend/conversations and backend/logs/conversations directories.
        conv_files = sorted([f for f in filtered_files if f.startswith(conv_prefix)])
        log_conv_files = sorted([f for f in filtered_files if f.startswith(log_conv_prefix)])
        
        # Remove these from the general list.
        filtered_files = [f for f in filtered_files if not (f.startswith(conv_prefix) or f.startswith(log_conv_prefix))]
        
        # For backend/conversations, choose only the first two and the last file
        if len(conv_files) >= 3:
            sample_conv = conv_files[:2] + [conv_files[-1]]
        else:
            sample_conv = conv_files
            
        # For .wav files, choose only the first two and the last file
        wav_files = sorted([f for f in filtered_files if f.endswith('.wav')])
        filtered_files = [f for f in filtered_files if not f.endswith('.wav')]
        
        if len(wav_files) >= 3:
            sample_wav = wav_files[:2] + [wav_files[-1]]
        else:
            sample_wav = wav_files
            
        # Add the sampled wav files back to filtered_files
        filtered_files.extend(sample_wav)
        
        # For each conversation file, get its base name (without extension) to match log files.
        sample_conv_ids = [os.path.splitext(os.path.basename(f))[0] for f in sample_conv]
        
        # For backend/logs/conversations, pick only files whose base name matches one of the sampled conv files.
        sample_logconv = []
        for conv_id in sample_conv_ids:
            matched = next((f for f in log_conv_files if os.path.splitext(os.path.basename(f))[0] == conv_id), None)
            if matched:
                sample_logconv.append(matched)
        
        # Final file list: all other files + sampled conversations + sampled log conversation files.
        final_list = filtered_files + sample_conv + sample_logconv
        
        # It's up to you how to order the final output.
        # Here we sort them alphabetically.
        final_list = sorted(final_list)
        
        # Write the final list to the output file (created in the get_code folder)
        output_path = os.path.join(script_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            for file in final_list:
                f.write(file + "\n")
        
        print(f"Created file list at: {output_path}")
        print(f"Total files listed: {len(final_list)}")
        
    except Exception as e:
        print(f"Error creating file list: {e}")
        sys.exit(1)

if __name__ == "__main__":
    create_concat_list()