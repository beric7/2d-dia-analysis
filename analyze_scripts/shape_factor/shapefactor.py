import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
    return (4 / 3) * np.pi * (dim1 * dim2 * dim3) / (2**3)

# Function to calculate cumulative volume distribution
def calculate_cumulative_volume(results_df, sieve_bins):
    # Group grains by true minimum dimension (sieve bins)
    results_df['sieve_bin'] = pd.cut(results_df['true_min_dim'], bins=sieve_bins, labels=False)
    
    # Calculate total volume for each bin
    bin_volumes = results_df.groupby('sieve_bin')['volume'].sum()
    
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
def derive_realistic_bounds(dim1, dim2, population_data, aspect_ratio_bin):
    # Filter population data for the given aspect ratio bin
    filtered_data = population_data[population_data['aspect_ratio_bin'] == aspect_ratio_bin]
    
    # Further filter for grains with similar dim1 and dim2 values
    filtered_data = filtered_data[
        (filtered_data['feret_min_mm'] <= dim1) & (filtered_data['feret_max_mm'] >= dim2)
    ]
    
    # Debugging: Check if filtered_data is empty
    if filtered_data.empty:
        print(f"Empty filtered data for dim1={dim1}, dim2={dim2}, aspect_ratio_bin={aspect_ratio_bin}")
    
    # Calculate realistic bounds for dim3
    lower_bound = filtered_data['feret_min_mm'].min()
    upper_bound = filtered_data['feret_max_mm'].max()
    
    # Debugging: Print bounds
    # print(f"Bounds for dim1={dim1}, dim2={dim2}: lower_bound={lower_bound}, upper_bound={upper_bound}")
    
    return lower_bound, upper_bound

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
