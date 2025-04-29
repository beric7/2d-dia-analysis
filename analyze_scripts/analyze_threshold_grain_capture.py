import argparse
from tqdm import tqdm
import json
import time
import os
import pandas as pd
from position_priors import get_contour_df, load_json_data, get_gsd, get_percent_passing, get_curve
from json_grains_2_coco_json import list_files_full_path, plot_data, convert_to_coco
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor, as_completed
from calculate_sieve_results import get_results
from json_grains_2_coco_json import plot_data

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
    parser.add_argument('--img_dir', type=str, help='image directory')
    parser.add_argument('--save_dir', type=str, help='base save directory')
    parser.add_argument('--pixel2mm', type=float, help='PIXEL_TO_MM')
    parser.add_argument('--sand_name', type=str, help="options: " \
    "South_High, North_Low, North_High, South_Low, Virginia_Beach, Venice_Beach, OGT,Olivine, Washington_State")
    parser.add_argument('--exp', type=str, help='experiment name')
    args = parser.parse_args()

    img_dir = args.img_dir
    save_dir = args.save_dir
    pixel2mm = args.pixel2mm
    sand_name = args.sand_name
    experiment_name = args.exp
    json_save_dir = f'{save_dir}/individual_particles_json/'
    coco_json = f'{save_dir}/coco_formatted.json'
    save_grain_dir = f'{save_dir}/grains/'
    percent_values = [0.1, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.84, 0.9]

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(save_grain_dir):
        os.mkdir(save_grain_dir)
    if not os.path.exists(json_save_dir):
        os.mkdir(json_save_dir)

    start = time.time()
    # List all files in the directory
    files = os.listdir(img_dir)
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=24) as executor:
        # Submit tasks to the executor
        futures = {executor.submit(process_file, file, img_dir, save_grain_dir, pixel2mm, json_save_dir): file for file in files}
        # Use tqdm to display progress
        for future in tqdm(as_completed(futures), total=len(files)):
            future.result()  # This will raise any exceptions that occurred during processing
    end = time.time()
    print(f"Processing completed in {end - start:.2f} seconds.")
    print('listing fp')
    json_files = list_files_full_path(json_save_dir)  # List of JSON file paths
    image_paths = img_dir  # Corresponding full image paths
    coco_json = coco_json
    coco_data = convert_to_coco(json_files, image_paths)

    # Save the COCO formatted data to a file
    with open(coco_json, 'w') as f:
        json.dump(coco_data, f)

    # plot the histograms!
    plot_data(coco_json, save_dir)
    
    data = load_json_data(coco_json)
    data = data['annotations']
    df = pd.DataFrame(data)

    cs = get_gsd(df, key='ellip_volume')
    gsd_results = get_percent_passing(cs, save_dir, percent_values=percent_values)
    # save GSD curve
    df_gsd = get_curve(df, save_dir, key='ellip_volume')
    df_gsd.to_csv(f'{save_dir}/save_annotations_df_{sand_name}.csv')

    comparison_df = pd.read_csv(f'/projects/OLIVINE/data/sieve_results/{sand_name}/{sand_name}.csv')
    comparison_df[experiment_name] = gsd_results['Size']
    comparison_df.to_csv(f'/projects/OLIVINE/data/sieve_results/{sand_name}/{experiment_name}.csv', index=False)

if __name__ == "__main__":
    main()