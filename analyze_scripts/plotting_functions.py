# Copyright 2025 The MITRE Corporation

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.integrate import trapezoid   # For numerical integration
from scipy.stats import norm  # For Z-scores
from utils.load_json import load_json_data
import math
from typing import Any, Dict, List, Tuple

def extract_data(coco_format: Any) -> Tuple[List[float], List[float], List[float], List[float], List[bool], List[float], List[float]]:
    """
    Extracts particle attributes from COCO-format annotation data.

    Args:
        coco_format (dict or list): COCO-format annotation data.

    Returns:
        -> Tuple of lists: (areas, feret_maxes_mm, feret_mins_mm, sphericities, touch_edges, volumes, ellip_volumes)
    """
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
    except Exception:
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

def get_gsd(df: pd.DataFrame, key: str = 'volume') -> List[Tuple[float, float, float]]:
    """
    Compute the grain size distribution (GSD) as a cumulative sum and normalize.

    Args:
        df (pd.DataFrame): DataFrame containing particle data.
        key (str): Column to use for volume (default 'volume').

    Returns:
        -> List[Tuple[float, float, float]]: List of (feret_min_mm, volume, normalized cumulative sum).
    """
    runningSum = 0
    cumulativeSums = [(l, v, runningSum:=runningSum+v) for l, v in sorted(df[['feret_min_mm', key]].values, key=lambda x: x[0])]
    cumulativeSums = [(l, v, cs/cumulativeSums[-1][2]) for l, v, cs in cumulativeSums] # Normalize
    return cumulativeSums

def get_percent_passing(
    cumulativeSums: List[Tuple[float, float, float]],
    save_dir: str,
    percent_values: List[float] = [0.1, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.84, 0.9]
) -> pd.DataFrame:
    """
    Compute the particle size at specified percent passing values.

    Args:
        cumulativeSums (list): List of (feret_min_mm, volume, normalized cumulative sum).
        save_dir (str): Directory to save the CSV.
        percent_values (list): Percent passing values to compute.

    Returns:
        -> pd.DataFrame: DataFrame with 'Percent Passing' and 'Size' columns.
    """
    checkpoint = None
    data = {'Percent Passing': [], 'Size': []}
    def check_pp(cs, fmin, percent_values, checkpoint):
        for percent in percent_values:
            if cs > percent and (checkpoint is None or checkpoint < percent * 100):
                checkpoint = percent * 100
                print(f'D{checkpoint}: {fmin}')
                data['Size'].append(fmin)
                data['Percent Passing'].append(checkpoint)
        return checkpoint

    for fmin, vol, cs in cumulativeSums:
        checkpoint = check_pp(cs, fmin, percent_values, checkpoint)

    df = pd.DataFrame(data)
    df.to_csv(f'{save_dir}/grain_size_distribution.csv', index=False)
    return df

def get_curve(df: pd.DataFrame, save_dir: str, key: str = 'volume') -> pd.DataFrame:
    """
    Generate and save the grain size distribution curve.

    Args:
        df (pd.DataFrame): DataFrame containing particle data.
        save_dir (str): Directory to save the plot.
        key (str): Column to use for volume (default 'volume').

    Returns:
        -> pd.DataFrame: DataFrame with cumulative volume and percentage passing.
    """
    df = df.sort_values(by='feret_min_mm')
    df['cumulative_volume'] = df[key].cumsum()
    total_volume = df[key].sum()
    df['percentage_passing'] = (df['cumulative_volume'] / total_volume) * 100

    plt.figure(figsize=(10, 6))
    plt.plot(df['feret_min_mm'], df['percentage_passing'], marker='o', linestyle='-')
    plt.xlabel('Particle Size (mm)')
    plt.ylabel('Percentage Passing (%)')
    plt.title('Grain Size Distribution')
    plt.grid(True)
    plt.savefig(f'{save_dir}/grain_size_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    return df

def create_histograms(
    areas: List[float],
    feret_maxes: List[float],
    feret_mins: List[float],
    sphericities: List[float],
    touch_edges: List[bool],
    volumes: List[float],
    ellip_volumes: List[float],
    save_dir: str,
    use_heatmap: bool = False
) -> None:
    """
    Create and save histograms (and optionally heatmaps) for particle attributes.

    Args:
        areas, feret_maxes, feret_mins, sphericities, touch_edges, volumes, ellip_volumes: Lists of particle attributes.
        save_dir (str): Directory to save the figure.
        use_heatmap (bool): Whether to use heatmaps for the last two subplots.

    Returns:
        -> None
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

    # Define bin edges for 30 bins between 0 and 1
    bin_edges = np.linspace(0, 1, 31) # 30 bins between 0 and 1
    # Convert ellip_volumes to a NumPy array
    ellip_volumes = np.array(ellip_volumes)

    # Calculate histogram manually using numpy
    hist, edges = np.histogram(ellip_volumes, bins=bin_edges)

    # Handle overflow bin (values greater than 1)
    overflow = np.sum(ellip_volumes > 1) # Count values above 1

    # Plot bar chart for histogram
    axs[1, 1].bar(edges[:-1], hist, width=np.diff(edges), align='edge', color='orange', alpha=0.7, label='Data')
    # Add overflow bin as a separate bar
    axs[1, 1].bar(1, overflow, width=0.05, color='red', alpha=0.7, label='Overflow (values > 1)')
    axs[1, 1].set_title('Ellipsoid Volumes')

    # Define bin edges for 30 bins between 0 and 1
    bin_edges = np.linspace(0, 1, 31) # 30 bins between 0 and 1
    # Convert ellip_volumes to a NumPy array
    volumes = np.array(volumes)

    # Calculate histogram manually using numpy
    hist, edges = np.histogram(volumes, bins=bin_edges)

    # Handle overflow bin (values greater than 1)
    overflow = np.sum(volumes > 1) # Count values above 1

    # Plot histogram using hist() function
    axs[1, 2].bar(edges[:-1], hist, width=np.diff(edges), align='edge', color='brown', alpha=0.7, label='Data')
    axs[1, 2].bar(1, overflow, width=0.05, color='red', alpha=0.7, label='Overflow (values > 1)')
    axs[1, 2].set_title('Volumes')

    # Calculate the aspect ratio
    aspect_ratios = [min_val / max_val for min_val, max_val in zip(feret_mins, feret_maxes)]

    if use_heatmap:
        # Smooth heatmap for Areas vs Aspect Ratio
        # sns.kdeplot(x=areas, y=aspect_ratios, ax=axs[2, 0], cmap="viridis", fill=True, bw_adjust=0.5)
        # axs[2, 0].hexbin(x=areas, y=aspect_ratios, gridsize=50, cmap="viridis") # `gridsize` controls resolution
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
        axs[2, 1].set_title('Compactness')
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

def plot_aspectRatio_histograms_and_scatter(
    feret_mins: List[float],
    feret_maxes: List[float],
    save_dir: str,
    touch_edges: List[bool]
) -> None:
    """
    Plot aspect ratio histograms and scatter plots for different Feret min bins.

    Args:
        feret_mins (list): List of minimum Feret diameters.
        feret_maxes (list): List of maximum Feret diameters.
        save_dir (str): Directory to save the figure.
        touch_edges (list): List of touch edge flags.

    Returns:
        -> None
    """
    # Define the bins for Feret minimum values
    bins = [(0.01, 0.075), (0.075, 0.105), (0.105, 0.15), (0.15, 0.25), (0.25, 0.425), (0.425, 0.85)]
    titles = [
    'Aspect Ratio - 0.01mm to 0.075mm',
    'Aspect Ratio - 0.075mm to 0.105mm',
    'Aspect Ratio - 0.105mm to 0.15mm',
    'Aspect Ratio - 0.15mm to 0.25mm',
    'Aspect Ratio - 0.25mm to 0.425mm',
    'Aspect Ratio - 0.425mm to 0.85mm'
    ]

    # Create subplots: 2 rows (1 for histograms, 1 for scatter plots)
    fig, axs = plt.subplots(2, len(bins), figsize=(20, 10))
    fig.suptitle("Histograms and Scatter Plots of Particle Attributes")

    # Loop through each bin and calculate aspect ratios
    for i, (lower, upper) in enumerate(bins):
        # Filter data based on the current bin and exclude particles touching edges
        bin_indices = [j for j, min_val in enumerate(feret_mins) if lower <= min_val < upper and not touch_edges[j]]
        bin_aspect_ratios = [feret_mins[j] / feret_maxes[j] for j in bin_indices]
        bin_feret_mins = [feret_mins[j] for j in bin_indices]
        bin_feret_maxes = [feret_maxes[j] for j in bin_indices]

        # Plot histogram for the current bin
        axs[0, i].hist(bin_aspect_ratios, bins=30, color='blue', alpha=0.7)
        axs[0, i].set_title(titles[i])
        axs[0, i].set_xlabel('Aspect Ratio')
        axs[0, i].set_ylabel('Frequency')

        # Plot scatter plot for the current bin
        axs[1, i].scatter(bin_feret_mins, bin_feret_maxes, alpha=0.7, color='green')
        axs[1, i].set_title(f'Feret Min vs Max - {titles[i]}')
        axs[1, i].set_xlabel('Feret Min')
        axs[1, i].set_ylabel('Feret Max')

    plt.tight_layout()
    # Save the figure as a PNG file
    plt.savefig(f'{save_dir}/histogram_and_scatter_aspect_ratios.png')
    # plt.show()

def plot_data(json_file: str, save_dir: str, use_heatmap: bool = True) -> None:
    """
    Main function to extract data and generate all plots.

    Args:
        json_file (str): Path to the COCO-format JSON file.
        save_dir (str): Directory to save plots.
        use_heatmap (bool): Whether to use heatmaps in histograms.

    Returns:
        -> None
    """
    coco_format = load_json_data(json_file)
    areas, feret_maxes, feret_mins, sphericities, touch_edges, volumes, ellip_volumes = extract_data(coco_format)
    plot_aspectRatio_histograms_and_scatter(feret_mins, feret_maxes, save_dir, touch_edges)
    create_histograms(areas, feret_maxes, feret_mins, sphericities, touch_edges, volumes, ellip_volumes, save_dir, use_heatmap=use_heatmap)

def calculate_percentiles(
    feret_mins: List[float],
    percentages: List[float]
) -> Dict[str, float]:
    """
    Calculate percentile values for feret_min using normal distribution.

    Args:
        feret_mins (list): List of feret min values.
        percentages (list): List of percentiles (0-1).

    Returns:
        -> dict: Mapping of percentile label to value.
    """
    mean = np.mean(feret_mins)
    std_dev = np.std(feret_mins)
    percentile_values = {}
    for p in percentages:
        z_score = norm.ppf(p)
        value = mean + z_score * std_dev
        percentile_values[f"{int(p * 100)}% Passing"] = value
    return percentile_values

def calculate_area_error(
    ground_truth_df: pd.DataFrame,
    experiment_df: pd.DataFrame,
    experiment_name: str
) -> Tuple[float, List[float]]:
    """
    Calculate the absolute area error between ground truth and experiment curves.

    Args:
        ground_truth_df (pd.DataFrame): Ground truth data.
        experiment_df (pd.DataFrame): Experiment data.
        experiment_name (str): Name of the experiment column.

    Returns:
        -> Tuple[float, List[float]]: (absolute area error, list of errors at each point)
    """
    sizes = np.linspace(ground_truth_df['ground truth size'].min(), 
                        ground_truth_df['ground truth size'].max(), 500)
    ground_truth_interp = np.interp(sizes, ground_truth_df['ground truth size'], ground_truth_df['percentage passing'])
    experiment_interp = np.interp(sizes, experiment_df[experiment_name], experiment_df['percentage passing'])

    errors = [abs(gt - pred) / gt for gt, pred in zip(ground_truth_df['ground truth size'], experiment_df[experiment_name])]
    print(errors)
    absolute_area_error = trapezoid(np.abs(ground_truth_interp - experiment_interp), sizes)
    return absolute_area_error, errors

def plot_experiment_data(
    directory: str,
    ground_truth_file: str,
    save_dir: str
) -> pd.DataFrame:
    """
    Plot ground truth and experiment data, compute and save error tables.

    Args:
        directory (str): Directory containing experiment CSV files.
        ground_truth_file (str): Path to ground truth CSV.
        save_dir (str): Directory to save plots and error tables.

    Returns:
        -> pd.DataFrame: DataFrame of absolute area errors for each experiment.
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
    fig, ax = plt.subplots(figsize=(8, len(error_data) * 0.5 + 1)) # Adjust height based on number of rows
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=error_df.values, colLabels=error_df.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.auto_set_column_width(col=list(range(len(error_df.columns)))) # Automatically adjust column widths
    error_table_png_path = os.path.join(save_dir, 'error_table.png')
    plt.savefig(error_table_png_path)
    plt.close()

    return error_df


