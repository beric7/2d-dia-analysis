# Copyright 2025 The MITRE Corporation

import argparse
import os
import pandas as pd
from contour_characteristics import get_contour_df
from utils.load_json import load_json_data
from plotting_functions import plot_data, get_gsd, get_percent_passing, get_curve
from process_amodal_predictions import process_json_files, combine_json_files

from plotting_functions import plot_experiment_data, calculate_percentiles, get_curve, get_gsd

def process_file(
    file: str,
    img_dir: str,
    save_grain_dir: str,
    pixel2mm: float,
    json_save_dir: str
) -> None:
    """
    Process a single image file to extract contour data and save as JSON.

    Args:
        file (str): Filename of the image to process.
        img_dir (str): Directory containing the image file.
        save_grain_dir (str): Directory to save grain data.
        pixel2mm (float): Conversion factor from pixels to millimeters.
        json_save_dir (str): Directory to save the output JSON file.

    Returns:
        -> None

    Output:
        Saves a JSON file containing contour data for the image.
    """
    filepath = os.path.join(img_dir, file)
    df = get_contour_df(filepath, save_grain_dir, pixel2mm)
    if df is not None:
        try:
            fp = os.path.join(json_save_dir, df['file_name'][0].split('.jpg')[0] + '.json')
            df.to_json(fp)
        except Exception as e:
            print(f"Error processing {df['file_name']}: {e}")

def main() -> None:
    """
    Main entry point for grain image analysis.

    Parses command-line arguments, processes prediction files, combines results,
    computes grain size distributions, generates plots, and saves outputs.

    Args:
        -> None

    Returns:
        -> None

    Output:
        - JSON files with particle data
        - CSV files with grain size distributions and percentiles
        - Plots comparing experiment and ground truth data
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='grain image analysis.')
    parser.add_argument('--pred_dir', type=str, help='prediction directory')
    parser.add_argument('--save_dir', type=str, help='base save directory')
    parser.add_argument('--pixel2mm', type=float, help='PIXEL_TO_MM')
    parser.add_argument('--sand_name', type=str, help="options: " \
    "South_High, North_Low, North_High, South_Low, Virginia_Beach, Venice_Beach, OGT, Olivine, Washington_State")
    parser.add_argument('--exp', type=str, help='experiment name')
    args = parser.parse_args()

    # Assign command-line arguments to variables
    pred_dir: str = args.pred_dir
    save_dir: str = args.save_dir
    pixel2mm: float = args.pixel2mm
    sand_name: str = args.sand_name
    experiment_name: str = args.exp

    # Define directories and filenames for saving outputs
    json_save_dir: str = f'{save_dir}/particles_json/'
    coco_json: str = f'{save_dir}/coco_formatted.json'
    percent_values: list[float] = [0.01, 0.05, 0.1, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.84, 0.9, 0.95, 0.99]

    # Create output directories if they do not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(json_save_dir):
        os.makedirs(json_save_dir)

    # Process all prediction JSON files and combine them into a single COCO-style JSON
    process_json_files(pred_dir, json_save_dir, pixel2mm)
    combine_json_files(json_save_dir, coco_json)
    
    # Load the combined JSON data
    data: list[dict] = load_json_data(coco_json)

    # Convert loaded data to a pandas DataFrame for analysis
    df: pd.DataFrame = pd.DataFrame(data)
    # Optionally filter out grains touching the edge
    # df = df[df['touch_edge'] == False]

    # Calculate grain size distribution (GSD) using the 'ellip_volume' key
    cs = get_gsd(df, key='ellip_volume')
    gsd_results = get_percent_passing(cs, save_dir, percent_values=percent_values)

    # Plot histograms and heatmaps of the data
    plot_data(coco_json, save_dir, use_heatmap=True)
    
    # Save the GSD curve as a CSV file
    df_gsd = get_curve(df, save_dir, key='ellip_volume')
    df_gsd.to_csv(f'{save_dir}/save_annotations_df_{sand_name}.csv')

    # Load ground truth CSV for comparison and add experiment results
    gt_csv: str = f'/projects/OLIVINE/data/sieve_results/{sand_name}/{sand_name}.csv'
    comparison_df: pd.DataFrame = pd.read_csv(gt_csv)
    comparison_df[experiment_name] = gsd_results['Size']
    graph_dir: str = f'/projects/OLIVINE/data/sieve_results/{sand_name}/'
    comparison_df.to_csv(f'/projects/OLIVINE/data/sieve_results/{sand_name}/{experiment_name}.csv', index=False)

    # Plot experiment data against ground truth
    plot_experiment_data(graph_dir, gt_csv, save_dir)

    # Calculate percentiles for the minimum Feret diameter and save as CSV
    feret_min_hist: dict = calculate_percentiles(df['feret_min_mm'], percent_values)
    # Convert the dictionary to a DataFrame
    feret_min_hist_df: pd.DataFrame = pd.DataFrame(list(feret_min_hist.items()), columns=["Percent Passing", "Value"])

    # Save the DataFrame as a CSV file
    feret_min_hist_df.to_csv(f'{save_dir}/feret_min_percentiles_{sand_name}.csv', index=False)

if __name__ == "__main__":
    main()
