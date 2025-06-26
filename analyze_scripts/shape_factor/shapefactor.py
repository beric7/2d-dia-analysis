import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Function to calculate shape factor based on aspect ratio and feretmin
def calculate_shape_factor(aspect_ratio, feretmin):
    # Example: Shape factor increases as aspect ratio approaches 1
    # Larger feretmin values bias shape factor toward round shapes
    if aspect_ratio > 1:
        aspect_ratio = 1 / aspect_ratio  # Normalize aspect ratio to be <= 1
    
    # Shape factor formula (adjust as needed)
    shape_factor = aspect_ratio + 0.1 * np.log(feretmin + 1)
    
    # Ensure shape factor is within a reasonable range
    shape_factor = np.clip(shape_factor, 0.5, 1.5)
    
    return shape_factor

# Function to calculate ellipsoidal volume
def calculate_volume(dim1, dim2, dim3):
    a = dim1 / 2
    b = dim2 / 2
    c = dim3 / 2

    volume = (4 / 3) * np.pi * a * b * c
    return volume

# Function to calculate cumulative volume distribution
def calculate_cumulative_volume(results_df, bins, save_dir, exp_name):
    # Group grains by true minimum dimension (sieve bins)
    results_df['sieve_bin'] = pd.cut(results_df['feret_min_mm'], bins=[0] + bins, labels=False, right=True)

    results_df.to_csv(f'{save_dir}/{exp_name}_results_df.csv')
    
    # Calculate total volume for each bin
    bin_volumes = results_df.groupby('sieve_bin')['volume'].sum()
    
    # Ensure all bins are represented (fill missing bins with 0 volume)
    bin_volumes = bin_volumes.reindex(range(len(bins)), fill_value=0)
    
    # Calculate cumulative volume as a percentage
    cumulative_volume = bin_volumes.cumsum() / bin_volumes.sum()
    
    return cumulative_volume

# Function to calculate the spherical volume
def calculate_spherical_volume(feretmin):
    """
    Calculate the volume of a sphere given the feretmin (diameter).
    """
    radius = feretmin / 2
    volume = (4 / 3) * np.pi * (radius**3)
    return volume

# Function to calculate aspect ratio
def calculate_aspect_ratio(dim1, dim2):
    return dim1 / dim2

# Function to assign aspect ratio bin
def assign_aspect_ratio_bin(aspect_ratio, bins):
    for i, (lower, upper) in enumerate(bins):
        if lower <= aspect_ratio < upper:
            return i
    return None  # If aspect ratio doesn't fit in any bin

# Function to derive realistic bounds for dim3
def derive_realistic_bounds(dim1, dim2, population_data):
    """
    Derive realistic bounds for dim3 based on grains with similar dim1 or dim2 values.
    """
    # # Filter population data for grains with similar dim1 or dim2 values
    # filtered_data = population_data[
    #     (population_data['feret_min_mm'] <= dim1 + 0.01) & (population_data['feret_min_mm'] >= dim1 - 0.01) |
    #     (population_data['feret_max_mm'] <= dim2 + 0.01) & (population_data['feret_max_mm'] >= dim2 - 0.01)
    # ]
    # Filter grains based on dim1 ± 0.01mm
    filtered_dim1 = population_data[
        (population_data['feret_min_mm'] >= dim1 - 0.01) &
        (population_data['feret_min_mm'] <= dim1 + 0.01)
    ]
    corresponding_dim2_from_dim1 = filtered_dim1['feret_max_mm'].values

    # Filter grains based on dim2 ± 0.01mm
    filtered_dim2 = population_data[
        (population_data['feret_max_mm'] >= dim2 - 0.01) &
        (population_data['feret_max_mm'] <= dim2 + 0.01)
    ]
    corresponding_dim1_from_dim2 = filtered_dim2['feret_min_mm'].values

    # Combine all possibilities for dim3
    all_possibilities = np.concatenate([corresponding_dim2_from_dim1, corresponding_dim1_from_dim2])

    # # Handle empty filtered data
    # if len(all_possibilities) == 0:
    #     print(f"Empty filtered data for dim1={dim1}, dim2={dim2}. Assigning default bounds.")
    #     return 0.01, 1.0  # Default bounds (adjust as needed)

    # Compute density function using Gaussian Kernel Density Estimation
    # kde = gaussian_kde(all_possibilities)
    # x_vals = np.linspace(min(all_possibilities), max(all_possibilities), 1000)
    # density = kde(x_vals)

    # # Derive realistic bounds (e.g., 95% confidence interval)
    # cumulative_density = np.cumsum(density) / np.sum(density)
    # lower_bound = x_vals[np.argmax(cumulative_density >= 0.025)]  # 2.5% lower bound
    # upper_bound = x_vals[np.argmax(cumulative_density >= 0.975)]  # 97.5% upper bound
    
    # # Calculate realistic bounds for dim3 based on dim1 and dim2
    lower_bound = filtered_dim1['feret_min_mm'].min()
    upper_bound = filtered_dim2['feret_max_mm'].max()

    return lower_bound, upper_bound

def get_dim3(dim1, dim2, population_data):
    # Filter grains based on dim1 ± 0.01mm
    filtered_dim1 = population_data[
        (population_data['feret_min_mm'] >= dim1 - 0.01) &
        (population_data['feret_min_mm'] <= dim1 + 0.01)
    ]
    corresponding_dim2_from_dim1 = filtered_dim1['feret_max_mm'].values

    # Filter grains based on dim2 ± 0.01mm
    filtered_dim2 = population_data[
        (population_data['feret_max_mm'] >= dim2 - 0.01) &
        (population_data['feret_max_mm'] <= dim2 + 0.01)
    ]
    corresponding_dim1_from_dim2 = filtered_dim2['feret_min_mm'].values

    # Combine all possibilities for dim3
    all_possibilities = np.concatenate([corresponding_dim2_from_dim1, corresponding_dim1_from_dim2])

    # Compute density function using Gaussian Kernel Density Estimation
    kde = gaussian_kde(all_possibilities)
    x_vals = np.linspace(min(all_possibilities), max(all_possibilities), 1000)
    density = kde(x_vals)

    # Compute the mean of the density function
    # The mean is calculated as the weighted average of x_vals using the density as weights
    mean_dim3 = np.sum(x_vals * density) / np.sum(density)

    return mean_dim3

# Function to estimate dim3 using modified ellipsoidal volume model
def estimate_dim3_with_shape_factor(dim1, dim2, volume, shape_factor, lower_bound, upper_bound):

    # Estimate dim3 using modified ellipsoidal volume formula
    dim3 = (volume * (3 / (4 * np.pi)) * (2**3)) / (dim1 * dim2 * shape_factor)
    
    # Restrict dim3 to realistic bounds
    dim3 = np.clip(dim3, lower_bound, upper_bound)
    
    return dim3

# Function to estimate dim3 using modified ellipsoidal volume model and spherical model
def estimate_dim3_with_spherical_model(dim1, dim2, volume, shape_factor, lower_bound, upper_bound):
    """
    Estimate dim3 using a spherical model for larger grains and an ellipsoidal model for smaller grains.
    """

    # Threshold for switching between spherical and ellipsoidal models
    spherical_threshold = 0.5  # Example: grains with feretmin > 0.5 mm are modeled as spheres
    
    if dim1 > spherical_threshold:
        # Use spherical model for larger grains
        dim3 = calculate_spherical_volume(dim1)
    else:
        # Use ellipsoidal model for smaller grains
        dim3 = (volume * (3 / (4 * np.pi)) * (2**3)) / (dim1 * dim2 * shape_factor)
    
    # Restrict dim3 to realistic bounds
    dim3 = np.clip(dim3, lower_bound, upper_bound)
    
    return dim3 
