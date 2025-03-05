#!/usr/bin/env python3
import os
import sys
import argparse
import logging
from tabulate import tabulate
import pandas as pd

# Import our dataset manager
from dataset_manager import SoundDatasetManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def list_classes(manager):
    """Display list of classes and their file counts."""
    summary = manager.get_dataset_summary()
    
    table_data = []
    for class_name, stats in summary["classes"].items():
        table_data.append([
            class_name, 
            stats["total"], 
            stats["included"],
            stats["total"] - stats["included"],
            f"{stats['included']/stats['total']*100:.1f}%" if stats["total"] > 0 else "0%"
        ])
    
    # Sort by class name
    table_data.sort(key=lambda x: x[0])
    
    # Add totals row
    table_data.append([
        "TOTAL",
        summary["total_files"],
        summary["included_files"],
        summary["excluded_files"],
        f"{summary['included_files']/summary['total_files']*100:.1f}%" if summary["total_files"] > 0 else "0%"
    ])
    
    headers = ["Class", "Total Files", "Included", "Excluded", "% Included"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

def list_files(manager, class_name=None, excluded_only=False, limit=None):
    """List files in the dataset, optionally filtered by class or exclusion status."""
    # Get data as DataFrame for easier filtering
    df = manager.export_to_dataframe()
    
    # Apply filters
    if class_name:
        df = df[df["class"] == class_name]
    
    if excluded_only:
        df = df[~df["include_in_training"]]
    
    # Apply limit if specified
    if limit and limit > 0:
        df = df.head(limit)
    
    # Format for display
    display_df = df[["path", "class", "include_in_training", "quality_rating", "notes"]]
    
    # Convert boolean to Yes/No
    display_df["include_in_training"] = display_df["include_in_training"].map({True: "Yes", False: "No"})
    
    # Rename columns for display
    display_df.columns = ["File Path", "Class", "Include", "Quality", "Notes"]
    
    print(tabulate(display_df, headers="keys", tablefmt="grid", showindex=False))
    print(f"\nShowing {len(display_df)} of {len(df)} files")

def exclude_files(manager, file_paths, reason=""):
    """Exclude specified files from training."""
    success_count = 0
    for path in file_paths:
        if manager.exclude_file(path, reason):
            success_count += 1
            print(f"Excluded: {path}")
        else:
            print(f"Failed to exclude: {path} (not found)")
    
    print(f"Successfully excluded {success_count} of {len(file_paths)} files")

def include_files(manager, file_paths):
    """Re-include specified files for training."""
    success_count = 0
    for path in file_paths:
        if manager.include_file(path):
            success_count += 1
            print(f"Re-included: {path}")
        else:
            print(f"Failed to include: {path} (not found)")
    
    print(f"Successfully included {success_count} of {len(file_paths)} files")

def set_quality(manager, file_path, rating):
    """Set quality rating for a file."""
    if manager.set_quality_rating(file_path, rating):
        print(f"Set quality rating {rating} for {file_path}")
    else:
        print(f"Failed to set quality rating for {file_path} (not found)")

def export_to_csv(manager, output_file):
    """Export dataset metadata to CSV file."""
    df = manager.export_to_dataframe()
    df.to_csv(output_file, index=False)
    print(f"Exported {len(df)} records to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Manage sound dataset")
    parser.add_argument('--data-dir', default='data', help='Base data directory')
    parser.add_argument('--sounds-dir', default='sounds/training_sounds', help='Sound files directory (relative to data-dir)')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Scan command
    scan_parser = subparsers.add_parser('scan', help='Scan directory and update metadata')
    
    # List classes command
    list_classes_parser = subparsers.add_parser('classes', help='List all classes and their file counts')
    
    # List files command
    list_files_parser = subparsers.add_parser('files', help='List files in the dataset')
    list_files_parser.add_argument('--class', dest='class_name', help='Filter by class')
    list_files_parser.add_argument('--excluded', action='store_true', help='Show only excluded files')
    list_files_parser.add_argument('--limit', type=int, help='Limit number of files shown')
    
    # Exclude files command
    exclude_parser = subparsers.add_parser('exclude', help='Exclude files from training')
    exclude_parser.add_argument('file_paths', nargs='+', help='File paths to exclude (relative to class directory)')
    exclude_parser.add_argument('--reason', help='Reason for exclusion')
    
    # Include files command
    include_parser = subparsers.add_parser('include', help='Re-include files for training')
    include_parser.add_argument('file_paths', nargs='+', help='File paths to include (relative to class directory)')
    
    # Set quality command
    quality_parser = subparsers.add_parser('quality', help='Set quality rating for files')
    quality_parser.add_argument('file_path', help='File path (relative to class directory)')
    quality_parser.add_argument('rating', type=int, help='Quality rating (e.g., 1-5)')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export dataset metadata to CSV')
    export_parser.add_argument('output_file', help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Initialize dataset manager
    manager = SoundDatasetManager(
        data_dir=args.data_dir,
        sounds_dir=args.sounds_dir
    )
    
    # Process commands
    if args.command == 'scan':
        results = manager.scan_directory()
        print(f"Scanned directory and found {results['total_files']} files ({results['new_files']} new)")
    
    elif args.command == 'classes':
        list_classes(manager)
    
    elif args.command == 'files':
        list_files(manager, args.class_name, args.excluded, args.limit)
    
    elif args.command == 'exclude':
        exclude_files(manager, args.file_paths, args.reason or "")
    
    elif args.command == 'include':
        include_files(manager, args.file_paths)
    
    elif args.command == 'quality':
        set_quality(manager, args.file_path, args.rating)
    
    elif args.command == 'export':
        export_to_csv(manager, args.output_file)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 