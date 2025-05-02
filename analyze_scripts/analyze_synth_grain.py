import argparse
import os
import pandas as pd
from contour_characteristics import load_json_data, get_gsd, get_percent_passing, get_curve
from process_amodal_predictions import process_annotations
from ..utils.json_grains_2_coco_json import plot_data

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='grain image analysis.')
    parser.add_argument('--coco_json', type=str, help='coco_json')
    parser.add_argument('--save_dir', type=str, help='base save directory')
    parser.add_argument('--pixel2mm', type=float, help='PIXEL_TO_MM')
    args = parser.parse_args()

    save_dir = args.save_dir
    pixel2mm = args.pixel2mm
    coco_json = args.coco_json

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    coco_json = load_json_data(coco_json)

    # plots the histograms!
    plot_data(coco_json, save_dir)
    
    processed_data = process_annotations(coco_json, pixel2mm)
    
    df = pd.DataFrame(processed_data)

    df.to_csv(f'{save_dir}/save_annotations_df.csv')
    cs = get_gsd(df, key='ellip_volume')
    get_curve(df, save_dir, key='ellip_volume')
    get_percent_passing(cs, save_dir)

if __name__ == "__main__":
    main()