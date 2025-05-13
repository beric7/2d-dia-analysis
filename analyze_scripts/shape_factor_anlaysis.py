from shape_factor.main import apply_shapefactor
from shape_factor.plots import plot_error_analysis_chart, plot_cumulative_volume_curve
import numpy as np

# Define Save Directories
save_dir = 'data/output/PARTICLE/Amodal/oryx/model_B/olivine_all_36fps_0.2/'
exp_name = 'olivine_all_36fps'

# Apply the shapefactor to the 2D DIA results
bins = [(0.01, 0.075), (0.075, 0.105), (0.105, 0.15), (0.15, 0.25), (0.25, 0.425), (0.425, 0.85), (0.85, 1.5)]
results_df = apply_shapefactor('./save_annotations_df_Olivine.csv', bins)

# Load sieve analysis cumulative volume data
sieve_data = "/projects/OLIVINE/data/sieve_results/Olivine.csv"
cumulative_volume_sieve = list(np.asarray([0.05, 0.26, 24.52, 59.58, 79.44, 92.95, 99.99]) / 100)
sieve_bins = [0.01, 0.075, 0.105, 0.15, 0.25, 0.425, 0.85]

# Define our bins we will use for analysis (should match the sieve data, but includes an end cap)
bins_tested = [0.01, 0.075, 0.105, 0.15, 0.25, 0.425, 0.85, 1.5] # add an end cap to complete the curve.

# Calculate cumulative volume distribution
cumulative_volume_estimated = plot_cumulative_volume_curve(bins_tested, sieve_data, sieve_bins, cumulative_volume_sieve, results_df)

# plot error analysis
plot_error_analysis_chart(cumulative_volume_estimated, cumulative_volume_sieve, sieve_bins, save_dir, exp_name)
