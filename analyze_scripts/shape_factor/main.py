from tqdm import tqdm
import pandas as pd
from .shapefactor import (
    assign_aspect_ratio_bin, calculate_aspect_ratio, derive_realistic_bounds,
    calculate_volume, calculate_shape_factor, estimate_dim3_with_shape_factor
)

# Main function for creating the results_df
def apply_shapefactor(pop_data, bins, save_dir, exp_name):
    # Load population data (example: feretmin, feretmax, dim3, aspect ratio bin)
    population_data = pd.read_csv(pop_data)

    # Iterate over grains in the dataset
    results = []
    for index, grain in tqdm(population_data.iterrows()):
        dim1 = grain['feret_min_mm']
        dim2 = grain['feret_max_mm']
        volume = grain['ellip_volume']  # Example: volume from sieve analysis
        
        # Derive realistic bounds for dim3
        lower_bound, upper_bound = derive_realistic_bounds(dim1, dim2, population_data)
        
        # Calculate aspect ratio and assign bin
        aspect_ratio = calculate_aspect_ratio(dim1, dim2)
        
        # Calculate shape factor based on aspect ratio and feretmin
        shape_factor = calculate_shape_factor(aspect_ratio, dim1)
        
        # Estimate dim3 using modified ellipsoidal volume model
        dim3_estimated = estimate_dim3_with_shape_factor(dim1, dim2, volume, shape_factor, lower_bound, upper_bound)
        
        # Calculate the true minimum dimension
        true_min_dim = [dim1, dim2, dim3_estimated]
        true_min_dim.sort()
        true_min_dim = true_min_dim[1]
        
        # Calculate volume using estimated dimensions
        estimated_volume = calculate_volume(dim1, dim2, dim3_estimated)
        
        # Append results
        results.append({
            'dim1': dim1,
            'dim2': dim2,
            'dim3_estimated': dim3_estimated,
            'true_min_dim': true_min_dim,
            'volume': estimated_volume,
            'shape_factor': shape_factor
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{save_dir}/{exp_name}_results_df.csv')

    return results_df
