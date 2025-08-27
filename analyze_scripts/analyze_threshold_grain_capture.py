# Copyright 2025 The MITRE Corporation

import argparse
from tqdm import tqdm
import json
import time
import os
import pandas as pd
from contour_characteristics import get_contour_df, load_json_data
from plotting_functions import get_percent_passing, get_curve, get_gsd, plot_data, plot_experiment_data, calculate_percentiles
from utils.json_grains_2_coco_json import list_files_full_path, convert_to_coco
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    Main entry point for batch grain image analysis.

    Parses command-line arguments, processes images in parallel,
    converts results to COCO format, generates plots, computes
    grain size distributions, and saves outputs.

    Args:
        -> None

    Returns:
        -> None

    Output:
        - JSON files with particle data (per image)
        - COCO-formatted JSON file
        - CSV files with grain size distributions and percentiles
        - Plots comparing experiment and ground truth data
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='grain image analysis.')
    parser.add_argument('--img_dir', type=str, help='image directory')
    parser.add_argument('--save_dir', type=str, help='base save directory')
    parser.add_argument('--pixel2mm', type=float, help='PIXEL_TO_MM')
    parser.add_argument('--sand_name', type=str, help="options: " \
        "South_High, North_Low, North_High, South_Low, Virginia_Beach, Venice_Beach, OGT, Olivine, Washington_State")
    parser.add_argument('--exp', type=str, help='experiment name')
    args = parser.parse_args()

    img_dir: str = args.img_dir
    save_dir: str = args.save_dir
    pixel2mm: float = args.pixel2mm
    sand_name: str = args.sand_name
    experiment_name: str = args.exp
    json_save_dir: str = f'{save_dir}/individual_particles_json/'
    coco_json: str = f'{save_dir}/coco_formatted.json'
    save_grain_dir: str = f'{save_dir}/grains/'
    percent_values: list[float] = [0.05, 0.1, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.84, 0.9, 0.95]

    # Ensure output directories exist
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(save_grain_dir):
        os.mkdir(save_grain_dir)
    if not os.path.exists(json_save_dir):
        os.mkdir(json_save_dir)

    start: float = time.time()
    # List all files in the image directory
    files: list[str] = os.listdir(img_dir)
    # Use ProcessPoolExecutor for parallel processing of images
    with ProcessPoolExecutor(max_workers=24) as executor:
        # Submit tasks to the executor
        futures = {executor.submit(process_file, file, img_dir, save_grain_dir, pixel2mm, json_save_dir): file for file in files}
        # Use tqdm to display progress
        for future in tqdm(as_completed(futures), total=len(files)):
            future.result()  # This will raise any exceptions that occurred during processing
    end: float = time.time()
    print(f"Processing completed in {end - start:.2f} seconds.")

    # List all JSON files created for individual particles
    json_files: list[str] = list_files_full_path(json_save_dir)
    image_paths: str = img_dir  # Directory containing the images

    # Convert individual JSON files to a single COCO-format JSON
    coco_data: dict = convert_to_coco(json_files, image_paths)
    with open(coco_json, 'w') as f:
        json.dump(coco_data, f)

    # Plot histograms and distributions
    plot_data(coco_json, save_dir)
    
    # Load COCO-format data and extract annotations
    data: dict = load_json_data(coco_json)
    annotations: list[dict] = data['annotations']
    df: pd.DataFrame = pd.DataFrame(annotations)

    # Compute grain size distribution (GSD) using 'ellip_volume'
    cs = get_gsd(df, key='ellip_volume')
    gsd_results = get_percent_passing(cs, save_dir, percent_values=percent_values)

    # Save GSD curve as CSV
    df_gsd = get_curve(df, save_dir, key='ellip_volume')
    df_gsd.to_csv(f'{save_dir}/save_annotations_df_{sand_name}.csv')

    # Load ground truth CSV for comparison and add experiment results
    gt_csv: str = f'/projects/OLIVINE/data/sieve_results/{sand_name}/{sand_name}.csv'
    comparison_df: pd.DataFrame = pd.read_csv(gt_csv)
    comparison_df[experiment_name] = gsd_results['Size']
    comparison_df.to_csv(f'/projects/OLIVINE/data/sieve_results/{sand_name}/{experiment_name}.csv', index=False)
    graph_dir: str = f'/projects/OLIVINE/data/sieve_results/{sand_name}/'

    # Plot experiment data against ground truth
    plot_experiment_data(graph_dir, gt_csv, save_dir)

    # Calculate percentiles for the minimum Feret diameter and save as CSV
    feret_min_hist: dict = calculate_percentiles(df['feret_min_mm'], percent_values)
    feret_min_hist_df: pd.DataFrame = pd.DataFrame(list(feret_min_hist.items()), columns=["Percent Passing", "Value"])
    feret_min_hist_df.to_csv(f'{save_dir}/feret_min_percentiles_{sand_name}.csv', index=False)

if __name__ == "__main__":
    main()
