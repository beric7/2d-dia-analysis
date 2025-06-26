from shape_factor.main import apply_shapefactor
from shape_factor.plots import plot_error_analysis_chart, plot_cumulative_volume_curve
import numpy as np
import pandas as pd
# Define Save Directories
save_dir = '/projects/OLIVINE/data/output/PARTICLE/Amodal/oryx/model_olivine/olivine_all_36fps_0.2'
exp_name = 'olivine_all_36fps'

# Apply the shapefactor to the 2D DIA results
bins = [0.075, 0.105, 0.15, 0.25, 0.425, 0.85]
# results_df = apply_shapefactor(f'{save_dir}/save_annotations_df_Olivine.csv', bins, save_dir, exp_name)
results_df = pd.read_csv('/projects/OLIVINE/data/output/PARTICLE/Otsu/oryx/venice_long1/venice_all_36fps_results_df.csv')
# Load sieve analysis cumulative volume data
sieve_data = "/projects/OLIVINE/data/sieve_results/Olivine.csv"
cumulative_volume_sieve = list(np.asarray([7.05, 20.56, 40.42, 75.48, 99.74, 99.95]) / 100) # Olivine
# cumulative_volume_sieve = list(np.asarray([0, 0.08188, 0.24904, 1.095, 13.76, 63.4, 98.26]) / 100)
# Calculate cumulative volume distribution
cumulative_volume_estimated = plot_cumulative_volume_curve(bins, cumulative_volume_sieve, results_df, save_dir, exp_name)

# plot error analysis
plot_error_analysis_chart(cumulative_volume_sieve, cumulative_volume_estimated, save_dir, exp_name)
