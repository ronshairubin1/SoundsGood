import os
import sys
from pathlib import Path

# Extensions for files that should be listed but not parsed
MEDIA_EXTENSIONS = {
    # Sound files
    '.wav', '.mp3', '.ogg', '.flac', '.m4a',
    # Image files
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'
}

def concatenate_files():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tree_file_path = os.path.join(script_dir, "code_tree.txt")
    files_list_path = os.path.join(script_dir, "files_to_concat.txt")
    output_path = os.path.join(script_dir, "combined_code.txt")
    
    try:
        # Verify critical files exist first
        if not os.path.exists(tree_file_path):
            raise FileNotFoundError(f"Tree file missing: {tree_file_path}")
        if not os.path.exists(files_list_path):
            raise FileNotFoundError(f"Files list missing: {files_list_path}")

        with open(output_path, "w", encoding="utf-8") as outfile:
            # Write tree file contents
            try:
                with open(tree_file_path, "r", encoding="utf-8") as tree_file:
                    tree_content = tree_file.read()
                    outfile.write(tree_content + "\n\n")
                    print(f"Wrote tree content ({len(tree_content)} characters)")
            except Exception as e:
                print(f"Failed to write tree: {str(e)}")
                raise

            # Process files from list
            try:
                with open(files_list_path, "r", encoding="utf-8") as f:
                    file_paths = [line.strip() for line in f if line.strip()]
                    print(f"Found {len(file_paths)} files to process")
                    
                    for idx, file_path in enumerate(file_paths, 1):
                        abs_path = os.path.join(os.path.dirname(script_dir), file_path)
                        print(f"Processing {idx}/{len(file_paths)}: {file_path}")
                        
                        # Write file header
                        outfile.write(f"### File: {file_path} ###\n\n")
                        
                        # Handle media files
                        ext = os.path.splitext(file_path)[1].lower()
                        if ext in MEDIA_EXTENSIONS:
                            media_type = "SOUND" if ext in {'.wav','.mp3'} else "IMAGE"
                            outfile.write("*"*80 + "\n")
                            outfile.write(f"{file_path}\n{media_type} FILE\n")
                            outfile.write("*"*80 + "\n\n")
                            continue
                            
                        # Handle code files
                        try:
                            with open(abs_path, "r", encoding="utf-8") as infile:
                                content = infile.read()
                                outfile.write(content)
                                outfile.write("\n" + "*"*80 + "\n\n")
                                print(f"Wrote {len(content)} bytes from {file_path}")
                        except Exception as e:
                            outfile.write(f"Error reading {file_path}: {str(e)}\n")
                            print(f"Error processing {file_path}: {str(e)}")
                            
            except Exception as e:
                print(f"Failed to process files: {str(e)}")
                raise

        print(f"Successfully created {output_path}")
        
    except Exception as e:
        print(f"Critical failure: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    concatenate_files() 