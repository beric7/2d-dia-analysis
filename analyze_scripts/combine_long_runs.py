# Copyright 2025 The MITRE Corporation

"""
Script to aggregate and analyze grain size distributions from multiple COCO-format JSON files,
remove outliers, compute grain size distribution (GSD) curves, and compare with ground truth.

Inputs:
    - Multiple COCO-format JSON files containing particle annotation data.
    - Ground truth CSV file for comparison.

Outputs:
    - CSV file with GSD curve.
    - Updated ground truth CSV with experiment results.

Workflow:
    1. Load annotation data from multiple JSON files.
    2. Concatenate and clean data (remove largest outliers).
    3. Compute GSD and percent passing.
    4. Save results and update ground truth comparison file.
"""

from utils.load_json import load_json_data
import pandas as pd
from plotting_functions import get_curve, get_gsd, get_percent_passing

# Directory to save results
save_dir = '/path/to/save/directory/'
# Name of the sand sample and experiment
sand_name = 'Triton'
experiment_name = 'Triton_amodal_144fps'
# Percent passing values for GSD calculation
percent_values = [0.01, 0.05, 0.1, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.84, 0.9, 0.95, 0.99]

# Load annotation data from three parts of the experiment
# /projects/SSC-IMAGE-STITCHING/OLIVINE/data/output/PARTICLE/Amodal/oryx/model_B/Triton/triton_part_1_144fps
d1 = load_json_data(f'/path/to/directory/triton_part_1_144fps/coco_formatted.json')
d2 = load_json_data(f'/path/to/directory/triton_part_2_144fps/coco_formatted.json')
d3 = load_json_data(f'/path/to/directory/triton_part_3_144fps/coco_formatted.json')

# Convert loaded data to DataFrames
df1 = pd.DataFrame(d1)
df2 = pd.DataFrame(d2)
df3 = pd.DataFrame(d3)

# Concatenate all data into a single DataFrame
df = pd.concat([df1, df2, df3])

# Sort by 'volume' and remove the largest 50 particles (assumed outliers)
df_sorted = df.sort_values(by='volume', ascending=False)
df = df_sorted.iloc[50:]  # Exclude largest volume particles

# Compute grain size distribution (GSD) using 'ellip_volume'
cs = get_gsd(df, key='ellip_volume')
gsd_results = get_percent_passing(cs, save_dir, percent_values=percent_values)

# Save GSD curve as CSV
df_gsd = get_curve(df, save_dir, key='ellip_volume')
df_gsd.to_csv(f'{save_dir}/save_annotations_df_{sand_name}.csv')

# Load ground truth CSV for comparison and add experiment results
comparison_df = pd.read_csv(f'/projects/OLIVINE/data/sieve_results/{sand_name}/{sand_name}.csv')
comparison_df[experiment_name] = gsd_results['Size']
comparison_df.to_csv(f'/projects/OLIVINE/data/sieve_results/{sand_name}/{experiment_name}.csv', index=False)
