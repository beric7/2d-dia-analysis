# Copyright 2025 The MITRE Corporation

import argparse
import os
import pandas as pd
from contour_characteristics import load_json_data, get_gsd, get_percent_passing, get_curve
from process_amodal_predictions import process_annotations
from utils.json_grains_2_coco_json import plot_data

def main() -> None:
    """
    Main entry point for grain image analysis from a COCO-format JSON.

    Parses command-line arguments, loads and processes annotation data,
    generates plots, computes grain size distributions, and saves results.

    Args:
        -> None

    Returns:
        -> None

    Output:
        - Plots of histograms and distributions (saved in save_dir)
        - CSV file with processed annotation data
        - CSV and other files with grain size distribution results
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='grain image analysis.')
    parser.add_argument('--coco_json', type=str, help='Path to COCO-format JSON file')
    parser.add_argument('--save_dir', type=str, help='Directory to save outputs')
    parser.add_argument('--pixel2mm', type=float, help='Pixel-to-millimeter conversion factor')
    args = parser.parse_args()

    save_dir: str = args.save_dir
    pixel2mm: float = args.pixel2mm
    coco_json_path: str = args.coco_json

    # Ensure the output directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Load annotation data from COCO-format JSON
    coco_json: list[dict] = load_json_data(coco_json_path)

    # Plot histograms and distributions
    plot_data(coco_json, save_dir)
    
    # Process annotation data (convert to mm, extract features, etc.)
    processed_data: list[dict] = process_annotations(coco_json, pixel2mm)
    
    # Convert processed data to DataFrame for further analysis
    df: pd.DataFrame = pd.DataFrame(processed_data)

    # Save processed annotation data as CSV
    df.to_csv(f'{save_dir}/save_annotations_df.csv', index=False)

    # Compute grain size distribution (GSD) using 'ellip_volume'
    cs = get_gsd(df, key='ellip_volume')

    # Save GSD curve and percent passing results
    get_curve(df, save_dir, key='ellip_volume')
    get_percent_passing(cs, save_dir)

if __name__ == "__main__":
    main()
