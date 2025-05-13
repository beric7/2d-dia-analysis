import argparse
from tqdm import tqdm
import json
import time
import os
import pandas as pd
from contour_characteristics import get_contour_df, load_json_data
from plotting_functions import plot_data, get_gsd, get_percent_passing, get_curve
from process_amodal_predictions import process_json_files, combine_json_files
from calculate_sieve_results import get_results
import matplotlib.pyplot as plt

import numpy as np

from plotting_functions import plot_experiment_data, calculate_percentiles, get_curve, get_gsd

def process_file(file, img_dir, save_grain_dir, pixel2mm, json_save_dir):
    filepath = os.path.join(img_dir, file)
    df = get_contour_df(filepath, save_grain_dir, pixel2mm)
    if df is not None:
        try:
            fp = os.path.join(json_save_dir, df['file_name'][0].split('.jpg')[0] + '.json')
            df.to_json(fp)
        except Exception as e:
            print(f"Error processing {df['file_name']}: {e}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='grain image analysis.')
    parser.add_argument('--pred_dir', type=str, help='prediction directory')
    parser.add_argument('--save_dir', type=str, help='base save directory')
    parser.add_argument('--pixel2mm', type=float, help='PIXEL_TO_MM')
    parser.add_argument('--sand_name', type=str, help="options: " \
    "South_High, North_Low, North_High, South_Low, Virginia_Beach, Venice_Beach, OGT,Olivine, Washington_State")
    parser.add_argument('--exp', type=str, help='experiment name')
    args = parser.parse_args()

    pred_dir = args.pred_dir
    save_dir = args.save_dir
    pixel2mm = args.pixel2mm
    sand_name = args.sand_name
    experiment_name = args.exp
    json_save_dir = f'{save_dir}/particles_json/'
    coco_json = f'{save_dir}/coco_formatted.json'
    percent_values = [0.1, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.84, 0.9]

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(json_save_dir):
        os.makedirs(json_save_dir)

    process_json_files(pred_dir, json_save_dir, pixel2mm)
    combine_json_files(json_save_dir, coco_json)
    
    data = load_json_data(coco_json)

    df = pd.DataFrame(data)
    # # df = df[df['touch_edge'] == False]
    cs = get_gsd(df, key='ellip_volume')
    gsd_results = get_percent_passing(cs, save_dir, percent_values=percent_values)

    # # plot the histograms!
    plot_data(coco_json, save_dir, use_heatmap=True)
    
    # # save GSD curve
    df_gsd = get_curve(df, save_dir, key='ellip_volume')
    df_gsd.to_csv(f'{save_dir}/save_annotations_df_{sand_name}.csv')

    gt_csv = f'/projects/OLIVINE/data/sieve_results/{sand_name}/{sand_name}.csv'
    comparison_df = pd.read_csv(gt_csv)
    comparison_df[experiment_name] = gsd_results['Size']
    graph_dir = f'/projects/OLIVINE/data/sieve_results/{sand_name}/'
    comparison_df.to_csv(f'/projects/OLIVINE/data/sieve_results/{sand_name}/{experiment_name}.csv', index=False)

    plot_experiment_data(graph_dir, gt_csv, save_dir)

    feret_min_hist = calculate_percentiles(df['feret_min_mm'], percent_values)
    # Convert the dictionary to a DataFrame
    feret_min_hist_df = pd.DataFrame(list(feret_min_hist.items()), columns=["Percent Passing", "Value"])

    # Save the DataFrame as a CSV file
    feret_min_hist_df.to_csv(f'{save_dir}/feret_min_percentiles_{sand_name}.csv', index=False)

if __name__ == "__main__":
    main()