#!/usr/bin/env python3
"""
Script to extract specific PNG files from multiple folders and rename them
with the folder name as prefix.

This script:
1. Searches through all subfolders in a specified directory
2. Looks for two specific files: "error_active_fingers.png" and "quicklook_active_fingers_with_force.png"
3. Copies these files to a "dump" folder
4. Renames them with the source folder name as prefix
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

def extract_and_rename_files(base_path):
    """
    Extract specific PNG files from subfolders and rename them with folder prefix.
    
    Args:
        base_path (str): Path to the main folder containing all the trial folders
    """
    
    # Convert to Path object for easier handling
    base_path = Path(base_path)
    
    # Check if base path exists
    if not base_path.exists():
        print(f"Error: The specified path '{base_path}' does not exist.")
        return
    
    # Create dump folder
    dump_folder = base_path / "dump"
    dump_folder.mkdir(exist_ok=True)
    print(f"Created/using dump folder: {dump_folder}")
    
    # Files to look for
    target_files = [
        "error_active_fingers.png",
        "quicklook_active_fingers_with_force.png"
    ]
    
    # Counters for tracking
    folders_processed = 0
    files_copied = 0
    missing_files = []
    
    # Iterate through all subdirectories
    for folder in base_path.iterdir():
        if folder.is_dir() and folder.name != "dump":
            folders_processed += 1
            folder_name = folder.name
            print(f"\nProcessing folder: {folder_name}")
            
            # Look for target files in this folder
            for target_file in target_files:
                source_file = folder / target_file
                
                if source_file.exists():
                    # Create new filename with folder prefix
                    new_filename = f"{folder_name}_{target_file}"
                    destination_file = dump_folder / new_filename
                    
                    # Copy file with new name
                    try:
                        shutil.copy2(source_file, destination_file)
                        print(f"  ✓ Copied: {target_file} -> {new_filename}")
                        files_copied += 1
                    except Exception as e:
                        print(f"  ✗ Error copying {target_file}: {e}")
                else:
                    print(f"  ⚠ Missing: {target_file}")
                    missing_files.append(f"{folder_name}/{target_file}")
    
    # Create a text file with all copied image names
    image_list_file = dump_folder / "copied_images_list.txt"
    copied_images = []
    
    # Collect all copied image names
    for image_file in dump_folder.glob("*.png"):
        copied_images.append(image_file.name)
    
    # Sort the list for better organization
    copied_images.sort()
    
    # Write to text file
    try:
        with open(image_list_file, 'w') as f:
            f.write("List of Copied Images\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total images: {len(copied_images)}\n")
            f.write("=" * 50 + "\n\n")
            
            # Group by interaction type for better readability
            interaction_types = {}
            for img in copied_images:
                # Extract interaction type from filename
                if img.startswith('hook_'):
                    interaction_type = 'Hook Interactions'
                elif img.startswith('pinch_'):
                    interaction_type = 'Pinch Interactions'
                elif img.startswith('power_grip_'):
                    interaction_type = 'Power Grip Interactions'
                else:
                    interaction_type = 'Other'
                
                if interaction_type not in interaction_types:
                    interaction_types[interaction_type] = []
                interaction_types[interaction_type].append(img)
            
            # Write grouped images
            for interaction_type, images in interaction_types.items():
                f.write(f"{interaction_type}:\n")
                f.write("-" * 30 + "\n")
                for img in images:
                    f.write(f"  {img}\n")
                f.write("\n")
            
            # Also write a simple list at the end
            f.write("Complete List (for LaTeX reference):\n")
            f.write("-" * 40 + "\n")
            for img in copied_images:
                f.write(f"{img}\n")
        
        print(f"Image list saved to: {image_list_file}")
        
    except Exception as e:
        print(f"Error writing image list file: {e}")
    
    # Summary
    print(f"\n" + "="*60)
    print(f"SUMMARY:")
    print(f"Folders processed: {folders_processed}")
    print(f"Files copied: {files_copied}")
    print(f"Missing files: {len(missing_files)}")
    
    if missing_files:
        print(f"\nFiles that were not found:")
        for missing in missing_files:
            print(f"  - {missing}")
    
    print(f"\nAll extracted files are now in: {dump_folder}")
    print(f"Image list file created: {image_list_file}")


def main():
    """
    Main function - hardcode your base path here
    """
    
    # HARDCODE YOUR PATH HERE
    # Replace this with the actual path to your folder containing all the trial folders
    BASE_PATH = r"C:/Users/Emanuel Wicki/Documents/MIT/Master Thesis/Figures/prep"
    
    # Alternative examples (uncomment and modify as needed):
    # BASE_PATH = r"/Users/username/path/to/trial/folders"  # macOS/Linux
    # BASE_PATH = r"D:\Research\Trial_Data"  # Windows
    
    print("File Extraction and Renaming Script")
    print("="*60)
    print(f"Base path: {BASE_PATH}")
    print(f"Looking for files: error_active_fingers.png, quicklook_active_fingers_with_force.png")
    print("="*60)
    
    # Run the extraction
    extract_and_rename_files(BASE_PATH)


if __name__ == "__main__":
    main()
