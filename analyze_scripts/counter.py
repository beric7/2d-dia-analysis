# Copyright 2025 The MITRE Corporation

import os
import json
from tqdm import tqdm
import pandas as pd
from analyze_scripts.utils.load_json import load_json_data

def count_images_and_rows(
    parent_directory: str,
    output_file: str,
    amodal: bool
) -> None:
    """
    Count the number of image JSON files and annotation rows in each experiment folder.

    Args:
        parent_directory (str): Path to the parent directory containing experiment folders.
        output_file (str): Path to the output JSON file for saving results.
        amodal (bool): If True, use 'particles_json' subfolder and raw JSON data;
                       if False, use 'individual_particles_json' and 'annotations' key.

    Returns:
        -> None

    Output:
        - Writes a JSON file mapping each folder to its image count and annotation row count.
        - Prints the row count for each folder during processing.
    """
    # Dictionary to store folder names, image counts, and row counts
    folder_data = {}

    # Get a list of all folders in the parent directory
    folders = [folder for folder in os.listdir(parent_directory) if os.path.isdir(os.path.join(parent_directory, folder))]

    # Use tqdm to show progress
    for folder_name in tqdm(sorted(folders), desc="Processing folders"):
        folder_path = os.path.join(parent_directory, folder_name)
        
        # Path to the subfolder containing particle JSON files
        if not amodal:
            subfolder_path = os.path.join(folder_path, "individual_particles_json")
        else:
            subfolder_path = os.path.join(folder_path, "particles_json")
        
        # Count the number of image JSON files in the subfolder
        if os.path.exists(subfolder_path) and os.path.isdir(subfolder_path):
            image_count = sum(
                1 for file_name in os.listdir(subfolder_path)
                if file_name.lower().endswith('.json')
            )
        else:
            image_count = 0

        try: 
            # Path to the "coco_formatted.json" file
            coco_file_path = os.path.join(folder_path, "coco_formatted.json")
            data = load_json_data(coco_file_path)
            if not amodal:
                data = data['annotations']
            # Load the JSON file into a pandas DataFrame
            df = pd.DataFrame(data)
            row_count = len(df)  # Get the number of rows in the DataFrame
        except Exception:
            row_count = 0
        print(row_count)

        # Add the folder name, image count, and row count to the dictionary
        folder_data[folder_name] = {
            "image_count": image_count,
            "row_count": row_count
        }

    # Write the dictionary to a JSON file
    with open(output_file, 'w') as file:
        json.dump(folder_data, file, indent=4)

if __name__ == "__main__":
    """
    Script entry point.

    Counts image JSON files and annotation rows in each experiment folder under the parent directory,
    and saves the results to a JSON file.

    Args:
        None (uses hardcoded paths below)

    Output:
        - JSON file with image and row counts per folder.
        - Prints row counts during processing.
    """
    # Specify the parent directory containing the folders
    parent_directory = "path/to/directory/"
    amodal = True
    # Specify the output file name
    output_file = "image_counts_amodal_exp_ssc_olivine.json"
    # Call the function
    count_images_and_rows(parent_directory, output_file, amodal)
    print(f"Image counts saved to {output_file}")
