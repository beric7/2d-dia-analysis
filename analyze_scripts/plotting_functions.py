import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from utils.load_json import load_json_data 

# Amodal Analysis
from scipy.integrate import trapz  # For numerical integration
from scipy.stats import norm  # For Z-scores

# ################################################
# Extract data for plotting
# ################################################

# extracts data for plotting
def extract_data(coco_format):
    areas = []
    feret_maxes = []
    feret_mins = []
    feret_maxes_mm = []
    feret_mins_mm = []
    sphericities = []
    touch_edges = []
    volumes = []
    ellip_volumes = []

    try:
        for annotation in coco_format['annotations']:
            if annotation['touch_edge'] == False:
                areas.append(annotation['area'])
                feret_maxes.append(annotation['feret_max'])
                feret_mins.append(annotation['feret_min'])
                feret_maxes_mm.append(annotation['feret_max_mm'])
                feret_mins_mm.append(annotation['feret_min_mm'])
                sphericities.append(annotation['sphericity'])
                touch_edges.append(annotation['touch_edge'])
                volumes.append(annotation['volume'])
                ellip_volumes.append(annotation['ellip_volume'])
    except:
        for annotation in coco_format:
            if annotation['touch_edge'] == False:
                areas.append(annotation['area'])
                feret_maxes.append(annotation['feret_max'])
                feret_mins.append(annotation['feret_min'])
                feret_maxes_mm.append(annotation['feret_max_mm'])
                feret_mins_mm.append(annotation['feret_min_mm'])
                sphericities.append(annotation['sphericity'])
                touch_edges.append(annotation['touch_edge'])
                volumes.append(annotation['volume'])
                ellip_volumes.append(annotation['ellip_volume'])        

    return areas, feret_maxes_mm, feret_mins_mm, sphericities, touch_edges, volumes, ellip_volumes

# ################################################
# GRAIN SIZE DISTRIBUTIONS
# ################################################

# get the GSD
def get_gsd(df, key='volume'):
    #Make a size distribution for Feret Min
    runningSum = 0
    #cumulativeSums = [(x, runningSum:=runningSum+x) for x in sorted(vaBeachDf['volume'].values)]
    cumulativeSums = [(l, v, runningSum:=runningSum+v) for l, v in sorted(df[['feret_min_mm', key]].values, key=lambda x: x[0])]
    cumulativeSums = [(l, v, cs/cumulativeSums[-1][2]) for l, v, cs in cumulativeSums] # Normalize the cumulative sums
    ls, vs, nvs = zip(*cumulativeSums) # Unpack the list of pairs into a pair of lists
    return cumulativeSums

# Get the percentage passing for an array of percent values
def get_percent_passing(cumulativeSums, save_dir, percent_values=[0.1, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.84, 0.9]):
    
    # Initialize checkpoint and data dictionary
    percent_values = percent_values
    checkpoint = None
    data = {'Percent Passing': [], 'Size': []}
    # Function to check and update percent passing
    def check_pp(cs, fmin, percent_values, checkpoint):
        for percent in percent_values:
            if cs > percent and (checkpoint is None or checkpoint < percent * 100):
                checkpoint = percent * 100
                print(f'D{checkpoint}: {fmin}')
                data['Size'].append(fmin)
                data['Percent Passing'].append(checkpoint)
        return checkpoint

    # Iterate over cumulative sums and apply the check_pp function
    for fmin, vol, cs in cumulativeSums:
        checkpoint = check_pp(cs, fmin, percent_values, checkpoint)

    # Create a DataFrame and save it as a CSV file
    df = pd.DataFrame(data)
    df.to_csv(f'{save_dir}/grain_size_distribution.csv', index=False)
    return df

# Get the grain size distribution curve!
# TODO seems repetative to the get_gsd... combine?
def get_curve(df, save_dir, key='volume'):
    # Sort the dataframe by feret_min
    df = df.sort_values(by='feret_min_mm')

    # Calculate the cumulative sum of estimated_volume_of_sand
    df['cumulative_volume'] = df[key].cumsum()

    # Convert cumulative volume to percentage passing
    total_volume = df[key].sum()
    df['percentage_passing'] = (df['cumulative_volume'] / total_volume) * 100

    # Plot the grain size distribution
    plt.figure(figsize=(10, 6))
    plt.plot(df['feret_min_mm'], df['percentage_passing'], marker='o', linestyle='-')
    plt.xlabel('Particle Size (mm)')
    plt.ylabel('Percentage Passing (%)')
    plt.title('Grain Size Distribution')
    plt.grid(True)

    # Save the figure
    plt.savefig(f'{save_dir}/grain_size_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    return df

# ################################################
# GRAIN CHARACTERISTIC ANALYSIS 
# ################################################

# creates Histograms of grain sample characteristics. 
def create_histograms(areas, feret_maxes, feret_mins, sphericities, touch_edges, volumes, ellip_volumes, save_dir, use_heatmap=False):
    """
    Create histograms and optionally smooth heatmaps for particle attributes.

    Parameters:
        areas (list): List of particle areas.
        feret_maxes (list): List of maximum Feret diameters.
        feret_mins (list): List of minimum Feret diameters.
        sphericities (list): List of sphericity values.
        touch_edges (list): List of touch edge flags.
        volumes (list): List of particle volumes.
        ellip_volumes (list): List of ellipsoid volumes.
        save_dir (str): Directory to save the figure.
        use_heatmap (bool): Whether to use heatmaps instead of scatter plots for the last two subplots.
    """
    # Create a figure and a set of subplots
    fig, axs = plt.subplots(3, 3, figsize=(10, 15))
    fig.suptitle("Histograms of Particle Attributes")

    # Plot each histogram
    axs[0, 0].hist(areas, bins=30, color='blue', alpha=0.7)
    axs[0, 0].set_title('Areas')

    axs[0, 1].hist(feret_maxes, bins=30, color='green', alpha=0.7)
    axs[0, 1].set_title('Feret Maxes')

    axs[0, 2].hist(feret_mins, bins=30, color='red', alpha=0.7)
    axs[0, 2].set_title('Feret Mins')

    axs[1, 0].hist(sphericities, bins=30, color='purple', alpha=0.7)
    axs[1, 0].set_title('Sphericities')

    axs[1, 1].hist(ellip_volumes, bins=30, color='orange', alpha=0.7)
    axs[1, 1].set_title('Ellipsoid Volumes')

    axs[1, 2].hist(volumes, bins=30, color='brown', alpha=0.7)
    axs[1, 2].set_title('Volumes')

    # Calculate the aspect ratio
    aspect_ratios = [min_val / max_val for min_val, max_val in zip(feret_mins, feret_maxes)]

    if use_heatmap:
        # Smooth heatmap for Areas vs Aspect Ratio
        # sns.kdeplot(x=areas, y=aspect_ratios, ax=axs[2, 0], cmap="viridis", fill=True, bw_adjust=0.5)
        # axs[2, 0].hexbin(x=areas, y=aspect_ratios, gridsize=50, cmap="viridis")  # `gridsize` controls resolution
        axs[2, 0].hist(aspect_ratios, bins=30, color='blue', alpha=0.7)
        axs[2, 0].set_title('Aspect Ratio')
        # axs[2, 0].set_title('Areas // Aspect Ratio (Heatmap)')
        # axs[2, 0].set_xlabel('Areas')
        # axs[2, 0].set_ylabel('Aspect Ratio')
    else:
        # Scatter plot for Areas vs Aspect Ratio
        axs[2, 0].scatter(areas, aspect_ratios, color='blue', alpha=0.7)
        axs[2, 0].set_title('Areas // Aspect Ratio (Scatter)')
        axs[2, 0].set_xlabel('Areas')
        axs[2, 0].set_ylabel('Aspect Ratio')

    # Calculate the compactness
    compactness = [((4 * area)**(1/2) / (math.pi)) / feret_max for area, feret_max in zip(areas, feret_maxes)]

    if use_heatmap:
        # # Smooth heatmap for Volumes vs Compactness
        # sns.kdeplot(x=volumes, y=compactness, ax=axs[2, 1], cmap="viridis", fill=True, bw_adjust=0.5)
        # axs[2, 1].hexbin(x=volumes, y=compactness, gridsize=50, cmap="viridis")
        axs[2, 1].hist(compactness, bins=30, color='green', alpha=0.7)
        axs[2, 1].set_title('Aspect Ratio')
        # axs[2, 1].set_title('Volumes // Compactness (Heatmap)')
        # axs[2, 1].set_xlabel('Volumes')
        # axs[2, 1].set_ylabel('Compactness')
    else:
        # Scatter plot for Volumes vs Compactness
        axs[2, 1].scatter(volumes, compactness, color='green', alpha=0.7)
        axs[2, 1].set_title('Volumes // Compactness (Scatter)')
        axs[2, 1].set_xlabel('Volumes')
        axs[2, 1].set_ylabel('Compactness')

    if use_heatmap:
        # Smooth heatmap for Volumes vs Compactness
        # sns.kdeplot(x=aspect_ratios, y=compactness, ax=axs[2, 2], cmap="viridis", fill=True, bw_adjust=0.5)
        axs[2, 2].hexbin(x=aspect_ratios, y=compactness, gridsize=50, cmap="viridis")
        axs[2, 2].set_title('aspect_ratios // Compactness (Heatmap)')
        axs[2, 2].set_xlabel('aspect_ratios')
        axs[2, 2].set_ylabel('Compactness')
    else:
        # Scatter plot for Volumes vs Compactness
        axs[2, 2].scatter(aspect_ratios, compactness, color='green', alpha=0.7)
        axs[2, 2].set_title('aspect_ratios // Compactness (Scatter)')
        axs[2, 2].set_xlabel('Aspect Ratios')
        axs[2, 2].set_ylabel('Compactness')

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the figure as a PNG file
    plt.savefig(f'{save_dir}/histogram.png')

# Main function to execute the plotting
def plot_data(json_file, save_dir, use_heatmap=True):
    coco_format = load_json_data(json_file)
    areas, feret_maxes, feret_mins, sphericities, touch_edges, volumes, ellip_volumes = extract_data(coco_format)
    create_histograms(areas, feret_maxes, feret_mins, sphericities, touch_edges, volumes, ellip_volumes, save_dir, use_heatmap=use_heatmap)

# ################################################
# Amodal Plots
# ################################################

def calculate_percentiles(feret_mins, percentages):
    """
    Calculates the median (50% passing) and values for specific percentages passing (0.1, 0.2, ..., 0.9).

    Parameters:
        feret_mins (list or np.array): List of feret min values.

    Returns:
        dict: Dictionary containing the median and values for specific percentages passing.
    """
    # Sort the data
    # feret_mins_sorted = np.sort(feret_mins)
    
    # Calculate mean and standard deviation
    mean = np.mean(feret_mins)
    std_dev = np.std(feret_mins)
    
    # Calculate percentiles using Z-scores
    percentile_values = {}
    for p in percentages:
        z_score = norm.ppf(p)  # Get Z-score for the percentile
        value = mean + z_score * std_dev  # Calculate the value using the normal distribution formula
        percentile_values[f"{int(p * 100)}% Passing"] = value
    
    return percentile_values

def calculate_area_error(ground_truth_df, experiment_df, experiment_name):
    """
    Calculates the absolute area error between the ground truth and experiment curves.

    Parameters:
        ground_truth_df (pd.DataFrame): Ground truth data containing 'percentage passing' and 'ground truth size'.
        experiment_df (pd.DataFrame): Experiment data containing 'percentage passing', 'ground truth size', and experiment column.
        experiment_name (str): Name of the experiment column in the experiment DataFrame.

    Returns:
        float: Absolute area error between the ground truth and experiment curves.
    """
    # Interpolate ground truth and experiment data to ensure alignment
    sizes = np.linspace(ground_truth_df['ground truth size'].min(), 
                        ground_truth_df['ground truth size'].max(), 500)
    ground_truth_interp = np.interp(sizes, ground_truth_df['ground truth size'], ground_truth_df['percentage passing'])
    experiment_interp = np.interp(sizes, experiment_df[experiment_name], experiment_df['percentage passing'])

    errors = [abs(gt - pred) / gt for gt, pred, in zip(ground_truth_df['ground truth size'], experiment_df[experiment_name])]
    print(errors)
    # Calculate the absolute area between the curves (error)
    absolute_area_error = trapz(np.abs(ground_truth_interp - experiment_interp), sizes)
    
    return absolute_area_error, errors

def plot_experiment_data(directory, ground_truth_file, save_dir):
    """
    Generates a line plot with ground truth and experiment data.

    Parameters:
        directory (str): Directory containing the experiment CSV files.
        ground_truth_file (str): Path to the ground truth CSV file.
        save_dir (str): Directory to save the plot.

    Returns:
        None
    """
    # Load ground truth data
    ground_truth_path = os.path.join(directory, ground_truth_file)
    ground_truth_df = pd.read_csv(ground_truth_path)
    
    # Ensure ground truth has the correct columns
    if not set(['percentage passing', 'ground truth size']).issubset(ground_truth_df.columns):
        raise ValueError("Ground truth file must contain 'percentage passing' and 'ground truth size' columns.")
    
    # Create a figure for the line plot
    plt.figure(figsize=(10, 6))
    
    # Plot ground truth data
    plt.plot(ground_truth_df['ground truth size'], ground_truth_df['percentage passing'], 
             label='Ground Truth', linewidth=2, color='black')
    
    # Initialize a list to store errors
    error_data = []
    error_percentage = []
    
    # Iterate through experiment files in the directory
    for file in os.listdir(directory):
        if file.endswith('.csv') and file != ground_truth_file:
            # Load experiment data
            experiment_path = os.path.join(directory, file)
            experiment_df = pd.read_csv(experiment_path)
            
            # Ensure experiment file has the correct columns
            if not set(['percentage passing', 'ground truth size']).issubset(experiment_df.columns):
                raise ValueError(f"Experiment file {file} must contain 'percentage passing' and 'ground truth size' columns.")
            
            # Extract experiment name (assumes it's the third column)
            try: 
                experiment_name = experiment_df.columns[2]
            except:
                continue
            
            # Plot experiment data
            plt.plot(experiment_df[experiment_name], experiment_df['percentage passing'], 
                     label=experiment_name, linewidth=1.5)
            
            # Calculate the error using the abstracted function
            absolute_area_error, errors = calculate_area_error(ground_truth_df, experiment_df, experiment_name)
            
            # Add error data to the list
            error_data.append({
                'Experiment': experiment_name,
                'Absolute Area Error': absolute_area_error
            })

            error_percentage.append({
                'Experiment': experiment_name,
                'Absolute Area Error': errors
            })
    
    # Finalize the plot
    plt.xlabel('Size')
    plt.xscale('log')
    plt.ylabel('Percentage Passing')
    plt.title('Ground Truth vs Experiment Data')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plot_path = os.path.join(save_dir, 'line_plot.png')
    plt.savefig(plot_path)
    plt.close()

    # Create error table
    error_df = pd.DataFrame(error_data)
    
    # Save the error table as a CSV file
    error_table_path = os.path.join(save_dir, 'error_table.csv')
    error_df.to_csv(error_table_path, index=False)

    error_percentages_df = pd.DataFrame(error_percentage)
    error_percentages_table_path = os.path.join(save_dir, 'error_percentage_table.csv')
    error_percentages_df.to_csv(error_percentages_table_path, index=False)

    # Save the error table as a PNG file
    fig, ax = plt.subplots(figsize=(8, len(error_data) * 0.5 + 1))  # Adjust height based on number of rows
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=error_df.values, colLabels=error_df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(error_df.columns))))  # Automatically adjust column widths
    error_table_png_path = os.path.join(save_dir, 'error_table.png')
    plt.savefig(error_table_png_path)
    plt.close()
    
    return error_df


