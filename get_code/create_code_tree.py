import os
from pathlib import Path
from typing import List, Set
from itertools import groupby
import sys

def generate_tree(
    start_path: str,
    output_file: str = "code_tree.txt",
    ignore_folders: Set[str] = None,
    ignore_patterns: Set[str] = None,
    ignore_extensions: Set[str] = None,
    summarize_folders: Set[str] = None,
    critical_folders: Set[str] = None,
    max_depth: int = None,
    file_sample_threshold: int = 3  # Show sample if more than this many similar files
) -> None:
    """
    Generate a tree structure of the codebase, excluding system and non-source files.
    
    Args:
        start_path: Root directory to start from
        output_file: Output file name
        ignore_patterns: Patterns to ignore in file/folder names
        ignore_extensions: File extensions to ignore
        summarize_folders: Folders to show without their contents
        critical_folders: Folders to always show all contents
        max_depth: Maximum depth to traverse (None for unlimited)
        file_sample_threshold: Number of similar files before showing sample
    """
    if ignore_patterns is None:
        ignore_patterns = {
            '.git', '__pycache__', 'node_modules', '.env',
            'venv', '.idea', '.vscode', 'build', 'dist',
            'egg-info', '.pytest_cache', '.mypy_cache',
            'coverage', '.coverage', '.tox', '.nox',
            'htmlcov', '.DS_Store'
        }
    
    if ignore_extensions is None:
        ignore_extensions = {
            # Compiled files
            '.pyc', '.pyo', '.pyd', '.so', '.dll', '.dylib',
            '.exe', '.obj', '.cache','.wav','.mp3','.ogg','.flac','.m4a',
            '.jpg','.jpeg','.png','.gif','.bmp','.tiff',
            # Package files
            '.whl', '.egg',
            
            # System files
            '.DS_Store',
            
            # IDE files
            '.iml', '.iws', '.ipr',
            
            # Temporary files
            '.tmp', '.temp', '.swp',
            
            # Build files
            '.o', '.a', '.lib'
        }
    
    if ignore_folders is None:
        ignore_folders = {
            'get_code'
        }

    if summarize_folders is None:
        summarize_folders = {
            'scripts', 'Documentation', 'docs', 'examples',
            'tests', 'test', 'samples'
        }
    
    if critical_folders is None:
        critical_folders = {
            'src', 'core', 'api', 'routers', 'models'
        }
    
    def should_include(name: str) -> bool:
        """Check if a file or directory should be included."""
        # Check the full name first
        if name in ignore_patterns or name == '.DS_Store':
            return False
        # Then check if it contains any of the patterns
        if any(pat in name for pat in ignore_patterns):
            return False
        # Finally check the extension
        ext = os.path.splitext(name)[1].lower()
        return ext not in ignore_extensions
    
    def group_similar_files(files: List[Path]) -> List[List[Path]]:
        """Group files that have similar patterns."""
        def get_file_pattern(filename: str) -> str:
            """Get pattern from filename to group similar files."""
            # Group by extension and general pattern
            parts = filename.split('-')
            if len(parts) > 1 and all(len(part) in [4, 8, 12] for part in parts[:-1]):
                # UUID-like pattern
                return 'uuid' + os.path.splitext(filename)[1]
            return os.path.splitext(filename)[1]
        
        # Sort files by their pattern
        sorted_files = sorted(files, key=lambda x: get_file_pattern(x.name))
        # Group files by pattern
        groups = [list(g) for _, g in groupby(sorted_files, key=lambda x: get_file_pattern(x.name))]
        return groups

    def get_tree_lines(path: str, prefix: str = "", depth: int = 0) -> List[str]:
        """Recursively generate tree lines."""
        if max_depth is not None and depth > max_depth:
            return []
        
        lines = []
        path_obj = Path(path)
        
        try:
            items = list(path_obj.iterdir())
            dirs = sorted([x for x in items if x.is_dir() and should_include(x.name)])
            files = [x for x in items if x.is_file() and should_include(x.name)]
            
            # Check if current directory is critical or has critical parent
            is_critical = (path_obj.name in critical_folders or
                         any(parent.name in critical_folders 
                             for parent in path_obj.parents))
            
            # Process directories first
            for i, item in enumerate(dirs):
                is_last = (i == len(dirs) - 1) and not files
                curr_prefix = "└── " if is_last else "├── "
                next_prefix = "    " if is_last else "│   "
                
                if item.name in summarize_folders and not is_critical:
                    item_count = sum(1 for _ in item.rglob('*') 
                                   if _.is_file() and should_include(_.name))
                    lines.append(f"{prefix}{curr_prefix}{item.name}/ ({item_count} files)")
                else:
                    lines.append(f"{prefix}{curr_prefix}{item.name}")
                    lines.extend(get_tree_lines(str(item), prefix + next_prefix, depth + 1))
            
            # Process files
            if is_critical:
                # Show all files in critical directories
                for i, item in enumerate(sorted(files)):
                    is_last = (i == len(files) - 1)
                    curr_prefix = "└── " if is_last else "├── "
                    lines.append(f"{prefix}{curr_prefix}{item.name}")
            else:
                # Use sampling for non-critical directories
                file_groups = group_similar_files(files)
                processed_files = []
                
                for group in file_groups:
                    if len(group) > file_sample_threshold:
                        processed_files.extend(group[:2] + [None] + [group[-1]])
                    else:
                        processed_files.extend(group)
                
                for i, item in enumerate(processed_files):
                    is_last = (i == len(processed_files) - 1)
                    curr_prefix = "└── " if is_last else "├── "
                    
                    if item is None:
                        lines.append(f"{prefix}│   ...")
                    else:
                        lines.append(f"{prefix}{curr_prefix}{item.name}")
                    
        except PermissionError:
            return [f"{prefix}[Permission Denied]"]
        
        return lines
    
    try:
        # Get the tree structure
        tree_lines = [".", ""]  # Start with root
        tree_lines.extend(get_tree_lines(start_path))
        
        # Get the root folder name
        root_folder = os.path.basename(start_path)
        
        # Write the tree to a file in the script's directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"{root_folder}\n")  # Write root folder name
            f.write("└── .\n")  # Connect root to tree
            # Indent the rest of the tree to align with the root
            for line in tree_lines:
                if line.strip():  # Skip empty lines
                    f.write("    " + line + "\n")
        
        # Print summary
        print(f"\nTree structure written to: {output_path}")
        print(f"Total lines: {len(tree_lines)}")
        
    except Exception as e:
        print(f"Error writing tree file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Use the parent directory of the script as the start path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    
    # Get parent folder name and create output filename
    parent_folder = os.path.basename(parent_dir)
    output_filename = f"code_tree.txt"
    
    # You can customize these sets for your specific needs
    custom_ignore_patterns = {
            '.git', '__pycache__', 'node_modules', '.env',
            'venv', '.idea', '.vscode', 'build', 'dist',
            'egg-info', '.pytest_cache', '.mypy_cache',
            'coverage', '.coverage', '.tox', '.nox',
            'htmlcov', '.DS_Store'
    }
    
    custom_ignore_extensions = {
            # Compiled files
            '.pyc', '.pyo', '.pyd', '.so', '.dll', '.dylib',
            '.exe', '.obj', '.cache','.wav','.mp3','.ogg','.flac','.m4a',
            '.jpg','.jpeg','.png','.gif','.bmp','.tiff',
            # Package files
            '.whl', '.egg',
            
            # System files
            '.DS_Store',
            
            # IDE files
            '.iml', '.iws', '.ipr',
            
            # Temporary files
            '.tmp', '.temp', '.swp',
            
            # Build files
            '.o', '.a', '.lib'
    }

    # Define folders to ignore
    custom_ignore_folders = {
        'get_code', 'scripts', 'Documentation', 'docs', 'examples',
            'tests', 'test', 'samples'
    }

    # Define folders to summarize
    custom_summarize_folders = {
        'models'
    }
    
    
    generate_tree(
        start_path=parent_dir,  # Changed to parent_dir
        output_file=output_filename,
        ignore_patterns=custom_ignore_patterns,
        ignore_folders=custom_ignore_folders,
        ignore_extensions=custom_ignore_extensions,
        summarize_folders=custom_summarize_folders,
        max_depth=None
    )