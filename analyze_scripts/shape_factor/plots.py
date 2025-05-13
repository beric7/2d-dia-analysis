import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from shapefactor import calculate_cumulative_volume

# Plot the GSD curve
def plot_cumulative_volume_curve(bins, sieve_data, sieve_bins, cumulative_volume_sieve, results_df, save_dir, exp_name):
    
    sieve_data = pd.read_csv(sieve_data)
    # Calculate cumulative volume distribution
    cumulative_volume_estimated = calculate_cumulative_volume(results_df, bins)

    # Plot cumulative volume distributions
    plt.plot(bins[:-1], cumulative_volume_estimated, label='Estimated')
    plt.plot(sieve_bins, cumulative_volume_sieve, label='Sieve Analysis')
    plt.xlabel('Feretmin (mm)')
    plt.xscale('log')
    plt.xlim(0.05, 0.85)
    plt.ylabel('Percent Passing')
    plt.legend()
    plt.show()  

    plt.savefig(f'{save_dir}/{exp_name}_gsd.png')

    return cumulative_volume_estimated

# Plot the error at each point along the GSD curve 10% - 100%
def plot_error_analysis_chart(cumulative_volume_estimated, cumulative_volume_sieve, sieve_bins, save_dir, exp_name):

    # Define target percent passing points (10%, 20%, ..., 100%)
    target_percent_passing = np.linspace(0.1, 1.0, 10)

    # Interpolate sieve analysis data (diameter vs. percent passing)
    sieve_interp_func = interp1d(cumulative_volume_sieve, sieve_bins, kind='linear', fill_value="extrapolate")

    # Interpolate estimated cumulative volume data (diameter vs. percent passing)
    estimated_interp_func = interp1d(cumulative_volume_estimated, sieve_bins[:-1], kind='linear', fill_value="extrapolate")

    # Get interpolated diameters at target percent passing points
    sieve_diameters_at_target = sieve_interp_func(target_percent_passing)
    estimated_diameters_at_target = estimated_interp_func(target_percent_passing)

    # Calculate percent errors
    percent_errors = [
        abs((estimated - sieve) / sieve) * 100
        for estimated, sieve in zip(estimated_diameters_at_target, sieve_diameters_at_target)
    ]

    # Print percent errors
    print("Percent Errors at Target Percent Passing Points:")
    for i, error in enumerate(percent_errors):
        print(f"Percent Passing {target_percent_passing[i] * 100:.0f}%: {error:.2f}%")

    # Plot percent errors
    plt.plot(target_percent_passing * 100, percent_errors, marker='o', label='Percent Error')
    plt.xlabel('Percent Passing (%)')
    plt.ylabel('Percent Error (%)')
    plt.title('Percent Error Between Estimated and Sieve Analysis')
    plt.legend()
    plt.show()

    plt.savefig(f'{save_dir}/{exp_name}_gsd_error.png')
