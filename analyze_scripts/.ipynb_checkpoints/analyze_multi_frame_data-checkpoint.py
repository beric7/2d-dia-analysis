import cv2
import os
from tqdm import tqdm
import sys

from object_tracking import (
    template_matching,
    detect_centroids_using_moments,
    combine_bidirectional_matches_with_vertical_priority,
    get_metadata,
    filter_metadata_with_iqr,
    analyze_and_plot_weighted_distances
)

from plotting_functions import plot_multiframe_data
from multiprocessing import Pool, cpu_count

def process_image_set(args):
    image_files, output_directory, i = args
    matches = []

    try:
        # Load three consecutive images
        f1, f2, f3 = image_files[i], image_files[i + 1], image_files[i + 2]
        image1 = cv2.imread(f1, cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(f2, cv2.IMREAD_GRAYSCALE)
        image3 = cv2.imread(f3, cv2.IMREAD_GRAYSCALE)

        # Perform template matching and detect centroids
        top_third_template1, matched_region1, max_loc1, offset1 = template_matching(image1, image2)
        top_third_template2, matched_region2, max_loc2, offset2 = template_matching(image2, image3)

        centroids_f1_2, color1 = detect_centroids_using_moments(top_third_template1, color=(255, 0, 0), threshold=80, visualize_binary=False)
        centroids_f2_1, color2 = detect_centroids_using_moments(matched_region1, color=(0, 255, 0), threshold=80, visualize_binary=False)
        centroids_f2_3, color3 = detect_centroids_using_moments(top_third_template2, color=(0, 255, 0), threshold=80, visualize_binary=False)
        centroids_f3_2, color3 = detect_centroids_using_moments(matched_region2, color=(0, 255, 0), threshold=80, visualize_binary=False)

        # Combine bidirectional matches with vertical priority
        matches_1_2 = combine_bidirectional_matches_with_vertical_priority(centroids_f1_2, centroids_f2_1)
        match_metadata_1_2 = get_metadata(matches_1_2, offset1)

        matches_2_3 = combine_bidirectional_matches_with_vertical_priority(centroids_f2_3, centroids_f3_2)
        match_metadata_2_3 = get_metadata(matches_2_3, offset2)

        filtered_metadata_1_2 = filter_metadata_with_iqr(match_metadata_1_2)
        filtered_metadata_2_3 = filter_metadata_with_iqr(match_metadata_2_3)

        # Generate a unique output filename based on the indices of the images
        output_filename = f"tracking_visualization_{i}_{i+1}_{i+2}.png"
        output_path = os.path.join(output_directory, output_filename)

        # Visualize tracking across three frames (optional)
        # visualize_tracking_three_frames_side_by_side(
        #     image1, image2, image3, filtered_metadata_1_2, filtered_metadata_2_3, output_path=output_path
        # )

        matches.extend(filtered_metadata_1_2)
        matches.extend(filtered_metadata_2_3)

    except Exception as e:
        print(f"Failed to process images {i}, {i+1}, {i+2}: {e}")

    return matches


def process_and_validate_tracking(directory, stride=1, output_directory="output", limit=500):
    """
    Processes three consecutive images from a directory with a given stride and validates tracking results.

    Args:
        directory (str): Path to the directory containing images.
        stride (int): Stride to move through the images in the directory.
        output_directory (str): Directory to save the visualization of tracking results.

    Returns:
        list: Combined matches from all processed image sets.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Get all image file paths from the directory
    image_files = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))])

    if (len(image_files) - 2) > limit:
        lim = limit
    else:
        lim = (len(image_files) - 2)

    # Prepare arguments for multiprocessing
    args_list = [(image_files, output_directory, i) for i in range(3, lim)]

    # Use multiprocessing to process image sets
    num_processes = cpu_count()
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_image_set, args_list), total=len(args_list)))

    # Combine matches from all processes
    combined_matches = []
    for matches in results:
        combined_matches.extend(matches)

    return combined_matches

# Example usage
sand = 'olivine'
sieve = 60
exp_name = f'{sand}_{sieve}'
filter = 200
save_dir = f'/projects/OLIVINE/data/sample_optical_flow/_{exp_name}_s{filter}/'
max_images_to_analyze = 500
directory_path = f'/projects/OLIVINE/data/input_device/Oryx/oryx/{sand}/{exp_name}_144fps/'
output_directory_path = os.path.join(save_dir, 'saved_optical_flow_images/')
matches = process_and_validate_tracking(directory_path, stride=1, output_directory=output_directory_path, limit=max_images_to_analyze)


analyze_and_plot_weighted_distances(matches, save_dir, exp_name)
plot_multiframe_data(matches, save_dir, exp_name, filter=filter)
