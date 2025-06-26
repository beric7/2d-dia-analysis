from scipy.optimize import minimize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize

def calculate_shape_factor(aspect_ratio, feret_min, feret_max, poly_coeffs):
    """
    Calculate the shape factor using a polynomial function with logarithmic terms.
    """
    shape_factor = (
        poly_coeffs[0]
        + poly_coeffs[1] * feret_min
    )
    
    # Ensure shape factor is within a reasonable range (e.g., 0.5 to 1.5)
    shape_factor = np.clip(shape_factor, 0.1, 10)
    return shape_factor

def calculate_volume_vectorized(dim1, dim2, dim3, aspect_ratio, poly_coeffs):
    """
    Vectorized calculation of volume for multiple grains using the polynomial shape factor.
    """
    a = dim1 / 2
    b = dim2 / 2
    c = dim3 / 2

    # Calculate shape factor using the polynomial
    shape_factor = calculate_shape_factor(aspect_ratio, dim1, dim2, poly_coeffs)

    # Calculate volume using the ellipsoidal model
    volume = (4 / 3) * np.pi * a * b * c * shape_factor
    return volume

def calculate_error(cumulative_volume_sieve, sieve_bins, cumulative_volume_estimated, bins_tested, return_full=False):
    """
    Calculate the error between the sieve curve and the estimated cumulative volume curve.
    If return_full=True, return the full tuple (percent_errors, sieve_diameters_at_target, estimated_diameters_at_target).
    Otherwise, return the Mean Squared Error as a scalar value.
    """
    # Define target percent passing points (10%, 20%, ..., 100%)
    target_percent_passing = np.linspace(0.1, 1.0, 10)

    # Interpolate sieve analysis data (percent passing -> diameter)
    sieve_interp_func = interp1d(cumulative_volume_sieve, sieve_bins, kind='linear', fill_value="extrapolate")

    # Interpolate estimated cumulative volume data (percent passing -> diameter)
    estimated_interp_func = interp1d(cumulative_volume_estimated, bins_tested[:-1], kind='linear', fill_value="extrapolate")

    # Get interpolated diameters at target percent passing points
    sieve_diameters_at_target = sieve_interp_func(target_percent_passing)
    estimated_diameters_at_target = estimated_interp_func(target_percent_passing)

    # Calculate percent errors
    percent_errors = [
        (((estimated - sieve) / sieve) * 100)
        for estimated, sieve in zip(estimated_diameters_at_target, sieve_diameters_at_target)
    ]

    # Return full tuple for plotting or scalar error for optimization
    if return_full:
        return percent_errors, sieve_diameters_at_target, estimated_diameters_at_target
    else:
        mse = np.mean(np.array(percent_errors)**2)
        return mse

def calculate_volume(dim1, dim2, dim3, model_type='ellipsoidal', shape_factor=1.0):
    """
    Calculate the volume of a grain based on the chosen model type and shape factor.
    """
    a = dim1 / 2
    b = dim2 / 2
    c = dim3 / 2

    if model_type == 'ellipsoidal':
        volume = (4 / 3) * np.pi * a * b * c * shape_factor
    elif model_type == 'platy':
        volume = (4 / 3) * np.pi * a * b * (c / 2) * shape_factor
    elif model_type == 'spherical':
        radius = dim1 / 2  # Assume dim1 is the diameter for spherical grains
        volume = (4 / 3) * np.pi * (radius**3) * shape_factor
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return volume

def optimize_parameters(cumulative_volume_sieve, sieve_bins, dia_data_df, initial_params):
    """
    Optimize polynomial coefficients to minimize the error between the sieve curve and the DIA data curve.
    Returns the optimized coefficients and the resultant DataFrame with dim3 estimation.
    """
    def objective_function(params):
        # Extract polynomial coefficients
        poly_coeffs = params

        # Calculate aspect ratio for each grain
        dia_data_df['aspect_ratio'] = dia_data_df['feret_min_mm'] / dia_data_df['feret_max_mm']

        # Estimate dim3 (e.g., average of feret_min and feret_max)
        dia_data_df['dim3'] = (dia_data_df['feret_min_mm'] + dia_data_df['feret_max_mm']) / 2

        # Calculate volume for each grain using the polynomial shape factor
        dia_data_df['volume'] = calculate_volume_vectorized(
            dia_data_df['feret_min_mm'],
            dia_data_df['feret_max_mm'],
            dia_data_df['dim3'],
            dia_data_df['aspect_ratio'],
            poly_coeffs
        )

        # Assign grains to sieve bins
        dia_data_df['sieve_bin'] = pd.cut(dia_data_df['feret_min_mm'], bins=sieve_bins, labels=False)

        # Calculate cumulative volume for DIA data
        bin_volumes = dia_data_df.groupby('sieve_bin')['volume'].sum()
        bin_volumes = bin_volumes.reindex(range(len(sieve_bins) - 1), fill_value=0)
        cumulative_volume_dia = bin_volumes.cumsum() / bin_volumes.sum()

        # Calculate error
        error = calculate_error(cumulative_volume_sieve, sieve_bins, cumulative_volume_dia, sieve_bins)
        print(f"Error: {error}, Parameters: {params}")  # Print error and parameters for each evaluation
        return error
    
    # Initial guesses for the polynomial coefficients
    num_coeffs = 2  # Example: 10 coefficients for a polynomial with logarithmic terms
    initial_params = [1.0] * num_coeffs  # Initial guesses for the coefficients

    # Optimize parameters using scipy's minimize function
    result = minimize(
        objective_function,
        initial_params,  # Initial guesses for the parameters
        method='Nelder-Mead',  # Optimization method
        options={'maxiter': 1000, 'disp': True}  # Display optimization progress
    )
    
    # Apply the optimized coefficients to the DataFrame
    optimized_coeffs = result.x

    # Calculate aspect ratio for each grain
    dia_data_df['aspect_ratio'] = dia_data_df['feret_min_mm'] / dia_data_df['feret_max_mm']

    # Estimate dim3 (e.g., average of feret_min and feret_max)
    dia_data_df['dim3'] = (dia_data_df['feret_min_mm'] + dia_data_df['feret_max_mm']) / 2

    # Calculate volume for each grain using the polynomial shape factor
    dia_data_df['volume'] = calculate_volume_vectorized(
        dia_data_df['feret_min_mm'],
        dia_data_df['feret_max_mm'],
        dia_data_df['dim3'],
        dia_data_df['aspect_ratio'],
        optimized_coeffs
    )

    # Assign grains to sieve bins
    dia_data_df['sieve_bin'] = pd.cut(dia_data_df['feret_min_mm'], bins=sieve_bins, labels=False)

    # Calculate cumulative volume for DIA data
    bin_volumes = dia_data_df.groupby('sieve_bin')['volume'].sum()
    bin_volumes = bin_volumes.reindex(range(len(sieve_bins) - 1), fill_value=0)
    cumulative_volume_dia = bin_volumes.cumsum() / bin_volumes.sum()

    return optimized_coeffs, dia_data_df, cumulative_volume_dia

def plot_error_analysis_chart(cumulative_volume_sieve, sieve_bins, cumulative_volume_estimated, bins_tested, save_dir, exp_name):
    """
    Plot the error at each point along the GSD curve (10% - 100% passing).
    """
    # Calculate percent errors and interpolated diameters
    percent_errors, sieve_diameters_at_target, estimated_diameters_at_target = calculate_error(
        cumulative_volume_sieve, sieve_bins, cumulative_volume_estimated, bins_tested, return_full=True
    )

    # Define target percent passing points (10%, 20%, ..., 100%)
    target_percent_passing = np.linspace(0.1, 1.0, 10)

    # Print percent errors
    print("Percent Errors at Target Percent Passing Points:")
    for i, error in enumerate(percent_errors):
        print(f"Percent Passing {target_percent_passing[i] * 100:.0f}%: {error:.2f}%, Sieve Diameter: {sieve_diameters_at_target[i]:.4f}, Estimated Diameter: {estimated_diameters_at_target[i]:.4f}")

    # Plot percent errors
    plt.figure(figsize=(10, 6))
    plt.plot(target_percent_passing * 100, percent_errors, marker='o', label='Percent Error')
    plt.xlabel('Percent Passing (%)')
    plt.ylabel('Percent Error (%)')
    plt.title('Percent Error Between Estimated and Sieve Analysis')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.8)  # Add a horizontal line at 0% error
    plt.legend()

    # Save the plot
    plt.savefig(f'{save_dir}/{exp_name}_gsd_error.png')
    plt.close('all')
save_dir = '/projects/OLIVINE/data/output/PARTICLE/Amodal/oryx/model_B/olivine_all_36fps_0.2'
exp_name = 'olivine_all_36fps'
sieve_data = "/projects/OLIVINE/data/sieve_results/Olivine.csv"
dia_data_df = pd.read_csv(f'{save_dir}/save_annotations_df_Olivine.csv')
cumulative_volume_sieve = list(np.asarray([0.05, 0.26, 24.52, 59.58, 79.44, 92.95, 99.99, 100]) / 100) # Olivine
# cumulative_volume_sieve = list(np.asarray([0, 0.08188, 0.24904, 1.095, 13.76, 63.4, 98.26, 100]) / 100)
sieve_bins = [0.01, 0.075, 0.105, 0.15, 0.25, 0.425, 0.85, 2]
# Optimize parameters
adjusted_cumulative_volume_sieve = cumulative_volume_sieve[1:]  # Remove the first value

# Optimize polynomial coefficients
optimized_coeffs, resultant_df, cumulative_volume_dia = optimize_parameters(
    cumulative_volume_sieve, sieve_bins, dia_data_df, initial_params=None
)

# Print optimized coefficients
print("Optimized Polynomial Coefficients:")
for i, coeff in enumerate(optimized_coeffs):
    print(f"Coefficient {i}: {coeff}")

# Save the resultant DataFrame
resultant_df.to_csv(f'{save_dir}/result_opt_df.csv')

# Plot error analysis chart
plot_error_analysis_chart(
    cumulative_volume_sieve, sieve_bins, cumulative_volume_dia, sieve_bins, save_dir, 'optimization_test'
)