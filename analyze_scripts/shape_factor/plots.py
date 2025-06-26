import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from .shapefactor import calculate_cumulative_volume

# Plot the GSD curve
def plot_cumulative_volume_curve(bins, cumulative_volume_sieve, results_df, save_dir, exp_name):
    # Calculate cumulative volume for the estimated data
    cumulative_volume_estimated = calculate_cumulative_volume(results_df, bins, save_dir, exp_name)
    
    # Plot the cumulative volume curve
    plt.figure(figsize=(8, 6))
    plt.plot(bins, cumulative_volume_sieve, label='Sieve Data', marker='x')
    plt.plot(bins, cumulative_volume_estimated, label='Estimated', marker='o')
    plt.xlabel('Sieve Bin Upper Bound')
    plt.ylabel('Cumulative Volume (%)')
    plt.title(f'Cumulative Volume Curve ({exp_name})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_dir}/{exp_name}_cumulative_volume_curve.png")
    plt.show()

    return cumulative_volume_estimated

# Plot the error at each point along the GSD curve 10% - 100%
def plot_error_analysis_chart(cumulative_volume_sieve, sieve_bins, cumulative_volume_estimated, bins_tested, save_dir, exp_name):
    # Define target percent passing points (10%, 20%, ..., 100%)
    target_percent_passing = np.linspace(0.1, 1.0, 10)

    # Interpolate sieve analysis data (diameter vs. percent passing)
    sieve_interp_func = interp1d(cumulative_volume_sieve, sieve_bins, kind='linear', fill_value="extrapolate")

    # Interpolate estimated cumulative volume data (diameter vs. percent passing)
    estimated_interp_func = interp1d(cumulative_volume_estimated, bins_tested[:-1], kind='linear', fill_value="extrapolate")

    # Get interpolated diameters at target percent passing points
    sieve_diameters_at_target = sieve_interp_func(target_percent_passing)
    estimated_diameters_at_target = estimated_interp_func(target_percent_passing)

    # Calculate percent errors
    percent_errors = [
        (((estimated - sieve) / sieve) * 100, sieve, estimated)
        for estimated, sieve in zip(estimated_diameters_at_target, sieve_diameters_at_target)
    ]

    # Print percent errors
    print("Percent Errors at Target Percent Passing Points:")
    for i, error in enumerate(percent_errors):
        print(f"Percent Passing {target_percent_passing[i] * 100:.0f}%: {error[0]:.2f}%, {error[1]}, {error[2]}")

    # Plot percent errors
    plt.plot(target_percent_passing * 100, percent_errors, marker='o', label='Percent Error')
    plt.xlabel('Percent Passing (%)')
    plt.ylabel('Percent Error (%)')
    plt.title('Percent Error Between Estimated and Sieve Analysis')
    plt.legend()

    plt.savefig(f'{save_dir}/{exp_name}_gsd_error.png')
    plt.close('all')
