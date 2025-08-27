# Copyright 2025 The MITRE Corporation

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import feret
from tqdm import tqdm
import os
from scipy.stats import mode

def template_matching(image1, image2):
    """
    Perform template matching to find the best match of the top third of the first image in the second image.

    Args:
        image1 (numpy.ndarray): First image (grayscale).
        image2 (numpy.ndarray): Second image (grayscale).

    Returns:
        tuple: Cropped sub-images from image1 and image2 for tracking.
    """
    # Extract the top third of the first image
    height, width = image1.shape
    top_template = image1[:int(height * (1/2)), :]  # Top 2/3 of image1
    bottom_template = image2

    # Perform template matching
    result = cv2.matchTemplate(bottom_template, top_template, cv2.TM_CCOEFF_NORMED)

    # Find the best match location
    _, _, _, max_loc = cv2.minMaxLoc(result)

    # Extract the best matching region from the second image
    match_height, match_width = top_template.shape
    offset = max_loc[1]
    matched_region = image2[offset :max_loc[1] + match_height, max_loc[0]:max_loc[0] + match_width]

    return top_template, matched_region, max_loc, offset

def detect_centroids_using_moments(image, color=(255, 0, 0), threshold=150, visualize_binary=False):
    """
    Detect centroids of all individual particles in an image using image moments.

    Args:
        image (numpy.ndarray): Input image (grayscale).
        threshold (int): Threshold value for binary segmentation.
        visualize_binary (bool): Whether to visualize the binary image.

    Returns:
        list: List of particle centroids [(x, y), ...].
    """
    # Apply binary thresholding
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    # Invert the image (make particles white and background black)
    inverted_image = cv2.bitwise_not(binary)
    color_image = np.zeros((inverted_image.shape[0], inverted_image.shape[1], 3), dtype=np.uint8)
    color_image[inverted_image == 255] = color

    # Visualize the binary image
    if visualize_binary:
        plt.figure(figsize=(8, 8))
        plt.imshow(inverted_image, cmap="gray")
        plt.title("Binary Image (Inverted)")
        plt.axis("off")
        plt.show()

    # Find contours
    contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    PIXEL_TO_MM = 0.00493177
    # Extract centroids for all individual particles using moments
    centroids = []
    for contour in contours:
        # Compute metrics
        feret_min, feret_max, area, volume = compute_particle_metrics(contour)
        # print(feret_min, feret_max, area, volume)
        # Calculate moments for each contour
        M = cv2.moments(contour)
        if M["m00"] != 0:  # Avoid division by zero
            cx = int(M["m10"] / M["m00"])  # x-coordinate of centroid
            cy = int(M["m01"] / M["m00"])  # y-coordinate of centroid
            centroids.append({'cx':cx, 'cy':cy, 
                              'area': area,
                              'fmin': feret_min,
                              'fmax': feret_max, 
                              'volume': volume})

    return centroids, color_image

def visualize_centroids(image, centroids, title="Detected Centroids"):
    """
    Visualize detected centroids of all individual particles on an image.

    Args:
        image (numpy.ndarray): Input image (RGB).
        centroids (list): List of centroids [(x, y), ...].
        title (str): Title for the visualization.
    """
    # Create a copy of the image for visualization
    image_with_centroids = image.copy()

    # Draw centroids on the image
    for centroid in centroids:
        x = centroid['cx']
        y = centroid['cy']
        cv2.circle(image_with_centroids, (x, y), 5, (255, 0, 0), -1)  # Draw a red circle for each centroid

    # Display the image with centroids
    plt.figure(figsize=(8, 8))
    plt.imshow(image_with_centroids)
    plt.title(title)
    plt.axis("off")
    plt.show()

def match_with_vertical_priority(centroids1, centroids2, reverse=False):
    """
    Match centroids between two sets with a bias toward vertical alignment.

    Args:
        centroids1 (list): List of centroids from the first image [(x, y), ...].
        centroids2 (list): List of centroids from the second image [(x, y), ...].

    Returns:
        list: List of matched centroid pairs [(x1, y1, x2, y2), ...].
    """
    matched = []
    for centroid_1 in centroids1:
        x1 = centroid_1['cx']
        y1 = centroid_1['cy']
        area1 = centroid_1['area']
        vol1 = centroid_1['volume']
        fmin1 = centroid_1['fmin']
        fmax1 = centroid_1['fmax']
        best_match = None
        best_score = float("inf")
        for centroid_2 in centroids2:
            x2 = centroid_2['cx']
            y2 = centroid_2['cy']
            area2 = centroid_2['area']
            vol2 = centroid_2['volume']
            fmin2 = centroid_2['fmin']
            fmax2 = centroid_2['fmax']
            # Calculate Euclidean distance
            euclidean_distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            
            # Introduce vertical bias: penalize horizontal displacement
            vertical_bias = abs(x2 - x1) * 2  # Weight horizontal distance more heavily
            
            # Combine distance and bias into a "score"
            score = euclidean_distance + vertical_bias
            
            # Update best match if this score is lower
            if score < best_score:
                best_score = score
                best_match = (x2, y2, area2, vol2, fmin2)
        
        # Append the best match for the current centroid
        if best_match:
            if reverse:
                matched.append({'match':(best_match[0], best_match[1], x1, y1), 
                                'biased_score': int(best_score), 
                                'v_dist': (y1 - best_match[1]), 
                                'h_dist': (x1 - best_match[0]),
                                'area1': area1,
                                'area2': area2,
                                'area_diff': (area1 - best_match[2]),
                                'vol_diff': (vol1 - best_match[3]),
                                'fmin_diff': (fmin1 - best_match[4]),
                                'fmin1': fmin1,
                                'fmin2': fmin2})
            else: 
                matched.append({'match':(x1, y1, best_match[0], best_match[1]), 
                                'biased_score': int(best_score), 
                                'v_dist': (y1 - best_match[1]), 
                                'h_dist': (x1- - best_match[0]),
                                'area_diff': (area1 - best_match[2]),
                                'area1': area1,
                                'area2': area2,
                                'vol_diff': (vol1 - best_match[3]),
                                'fmin_diff': (fmin1 - best_match[4]),
                                'fmin1': fmin1,
                                'fmin2': fmin2})
    return matched

def match_with_dynamic_weighting(centroids1, centroids2, features1, features2, size_threshold=50, weight_factor=2):
    """
    Match centroids between two sets with a bias toward vertical alignment, dynamically adjusting weights for smaller particles.

    Args:
        centroids1 (list): List of centroids from the first image [(x, y), ...].
        centroids2 (list): List of centroids from the second image [(x, y), ...].
        features1 (list): List of features for centroids1 [(area, volume, ...), ...].
        features2 (list): List of features for centroids2 [(area, volume, ...), ...].
        size_threshold (float): Threshold for particle size to adjust weighting.
        weight_factor (float): Factor to penalize horizontal displacement.

    Returns:
        list: List of matched centroid pairs [(x1, y1, x2, y2), ...].
    """
    matched = []
    for i, centroid_1 in enumerate(centroids1):
        x1 = centroid_1['cx']
        y1 = centroid_1['cy']
        best_match = None
        best_score = float("inf")
        area1, volume1 = features1[i]  # Extract features for the current particle

        for j, centroid_2 in enumerate(centroids2):
            x2 = centroid_2['cx']
            y2 = centroid_2['cy']
            area2, volume2 = features2[j]  # Extract features for the candidate match

            # Calculate Euclidean distance
            euclidean_distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Adjust vertical bias dynamically based on particle size
            if area1 < size_threshold or area2 < size_threshold:
                # Smaller particles get reduced horizontal penalty
                vertical_bias = abs(x2 - x1) * (weight_factor / 2)
            else:
                # Larger particles retain normal horizontal penalty
                vertical_bias = abs(x2 - x1) * weight_factor

            # Combine distance and bias into a "score"
            score = euclidean_distance + vertical_bias

            # Update best match if this score is lower
            if score < best_score:
                best_score = score
                best_match = (x2, y2)

        # Append the best match for the current centroid
        if best_match:
            matched.append((x1, y1, best_match[0], best_match[1]))
    return matched

def combine_bidirectional_matches_with_vertical_priority(centroids1, centroids2):
    """
    Combine matches from both forward (frame1 -> frame2) and reverse (frame2 -> frame1) directions,
    with vertical alignment bias.

    Args:
        centroids1 (list): List of centroids from the first image [(x, y), ...].
        centroids2 (list): List of centroids from the second image [(x, y), ...].

    Returns:
        list: Combined list of matched centroid pairs [(x1, y1, x2, y2), ...].
    """
    # Perform forward matching (frame1 -> frame2) with vertical bias
    forward_matches = match_with_vertical_priority(centroids1, centroids2)

    # Perform reverse matching (frame2 -> frame1) with vertical bias
    reverse_matches = match_with_vertical_priority(centroids2, centroids1, reverse=True)

    # Combine all matches into a single list
    combined_matches = forward_matches + reverse_matches

    return combined_matches

def visualize_tracking_with_color_scale(image1, image2, match_metadata, max_score=None):
    """
    Visualize particle tracking between two frames, stacking the images vertically, 
    and use a color scale (Jet colormap) to represent vertical distances of the match lines.

    Args:
        image1 (numpy.ndarray): First image (RGB).
        image2 (numpy.ndarray): Second image (RGB).
        matched_centroids (list): List of matched centroid pairs [(x1, y1, x2, y2), ...].
        match_metadata (list): List of match IDs and distances [(match_id, distance), ...].

    Returns:
        None
    """
    # Create a combined image for visualization (stacked vertically)
    combined_image = np.vstack((image1, image2))

    # Extract weighted scores for normalization
    weighted_scores = [value['score'] for value in match_metadata]
    if max_score is None:
        max_distance = max(weighted_scores)  # Maximum vertical distance for normalization
    else:
        max_distance = max_score
    # Normalize distances to [0, 1] range
    normalized_distances = [score / max_distance for score in weighted_scores]

    # Get Jet colormap
    # Get the colormap using the new method
    colormap = plt.colormaps["jet"]

    # Draw tracking lines and labels
    for i, value in enumerate(match_metadata):
        match_id = value['match_id']
        score = value['score']
        (x1, y1, x2, y2) = value['relative_centroid']
        _ = value['full_centroid']
        # Adjust the y-coordinate for the second image since it's stacked below the first image
        y2_adjusted = y2 + image1.shape[0]  # Add the height of the first image to y2

        # Normalize the vertical distance
        normalized_distance = normalized_distances[i]

        # Get color from Jet colormap (convert to BGR format for OpenCV)
        color = colormap(normalized_distance)[:3]  # RGB values
        color_bgr = tuple([int(255 * c) for c in reversed(color)])  # Convert to BGR for OpenCV

        # Draw a line connecting the centroids
        cv2.line(combined_image, (x1, y1), (x2, y2_adjusted), color_bgr, 2)
        cv2.circle(combined_image, (x1, y1), 5, (0, 255, 0), -1)  # Green circle for the first image
        cv2.circle(combined_image, (x2, y2_adjusted), 5, (0, 0, 255), -1)  # Blue circle for the second image

        # Add text label for the match ID
        label_position = ((x1 + x2) // 2, (y1 + y2_adjusted) // 2)  # Position the label midway along the trace
        cv2.putText(combined_image, match_id, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the result using matplotlib
    plt.figure(figsize=(8, 5), dpi=300)  # Adjust the figure size for vertical stacking
    plt.imshow(combined_image)
    # plt.colorbar(plt.cm.ScalarMappable(cmap=colormap), label="Weighted Distance")
    plt.title("Particle Tracking with Color Scale (Jet)")
    plt.axis("off")
    plt.savefig('Colorscale Tracking.png')

def visualize_tracking_with_color_scale_full_img(image1, image2, match_metadata, max_score=None):
    """
    Visualize particle tracking between two frames, stacking the images vertically, 
    and use a color scale (Jet colormap) to represent vertical distances of the match lines.

    Args:
        image1 (numpy.ndarray): First image (RGB).
        image2 (numpy.ndarray): Second image (RGB).
        matched_centroids (list): List of matched centroid pairs [(x1, y1, x2, y2), ...].
        match_metadata (list): List of match IDs and distances [(match_id, distance), ...].

    Returns:
        None
    """
    # Create a combined image for visualization (stacked vertically)
    combined_image = np.hstack((image1, image2))

    # Extract weighted scores for normalization
    weighted_scores = [value['score'] for value in match_metadata]
    if max_score is None:
        max_distance = max(weighted_scores)  # Maximum vertical distance for normalization
    else:
        max_distance = max_score

    # Normalize distances to [0, 1] range
    normalized_distances = [score / max_distance for score in weighted_scores]

    # Get Jet colormap
    # Get the colormap using the new method
    colormap = plt.colormaps["jet"]

    # Draw tracking lines and labels
    for i, value in enumerate(match_metadata):
        match_id = value['match_id']
        score = value['score']
        relative_centroid = value['relative_centroid']
        (x1, y1, x2, y2) = value['full_centroid']

        x2_adjusted = x2 + image1.shape[1]  # Add the height of the first image to y2

        # Normalize the vertical distance
        normalized_distance = normalized_distances[i]

        # Get color from Jet colormap (convert to BGR format for OpenCV)
        color = colormap(normalized_distance)[:3]  # RGB values
        color_bgr = tuple([int(255 * c) for c in reversed(color)])  # Convert to BGR for OpenCV

        # Draw a line connecting the centroids
        cv2.line(combined_image, (x1, y1), (x2_adjusted, y2), color_bgr, 2)
        cv2.circle(combined_image, (x1, y1), 5, (0, 255, 0), -1)  # Green circle for the first image
        cv2.circle(combined_image, (x2_adjusted, y2), 5, (0, 0, 255), -1)  # Blue circle for the second image

        # Add text label for the match ID
        label_position = ((x1 + x2_adjusted) // 2, (y1 + y2) // 2)  # Position the label midway along the trace
        cv2.putText(combined_image, match_id, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the result using matplotlib
    plt.figure(figsize=(15, 8), dpi=300)  # Adjust the figure size for vertical stacking
    # plt.imshow(combined_image)
    # plt.colorbar(plt.cm.ScalarMappable(cmap=colormap), label="Weighted Distance")
    plt.title("Particle Tracking with Color Scale (Jet)")
    plt.axis("off")
    plt.savefig('Colorscale Tracking.png')

def analyze_and_plot_weighted_distances(matches, save_dir, exp_name):
    """
    Analyze weighted distances for all matches, plot histograms, and display statistics.

    Args:
        matches (list): List of matched centroid pairs with metadata.
        save_dir (str): Directory to save the plots.
        exp_name (str): Experiment name for file naming.

    Returns:
        None
    """
    # Extract vertical distances and assign IDs
    weighted_scores = []
    v_dist = []
    h_dist = []
    for i, match in enumerate(matches):
        weighted_score = match['biased_score']
        v = match['v_dist']
        h = match['h_dist']
        
        # Save metadata
        v_dist.append(v)
        h_dist.append(h)
        weighted_scores.append(weighted_score)

    # Function to plot histogram with statistics
    def plot_histogram_with_statistics(data, bins, title, xlabel, ylabel, filename):
        plt.figure(figsize=(10, 6))
        bin_counts, bin_edges, patches = plt.hist(data, bins=bins, color='blue', alpha=0.7, edgecolor='black')
        plt.title(title, fontsize=16)
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)

        # Calculate statistics
        mean = np.mean(data)
        median = np.median(data)
        variance = np.var(data)
        std_dev = np.std(data)
        mode_value = mode(data).mode[0]
        mode_count = mode(data).count[0]

        # Display statistics below the histogram
        stats_text = (
            f"Mean: {mean:.2f}\n"
            f"Median: {median:.2f}\n"
            f"Variance: {variance:.2f}\n"
            f"Std Dev: {std_dev:.2f}\n"
            f"Mode: {mode_value:.2f} (Count: {mode_count})"
        )
        plt.gcf().text(0.85, 0.85, stats_text, fontsize=12, va='top', ha='right', bbox=dict(facecolor='white', alpha=0.8))

        # Save the plot
        plt.savefig(filename)
        plt.close()

    # Plot and save histograms
    plot_histogram_with_statistics(
        weighted_scores, bins=41,
        title="Histogram of Weighted Distances for Matches",
        xlabel="Weighted Distance",
        ylabel="Frequency",
        filename=f'{save_dir}/{exp_name}_Histogram_weighted_matches.png'
    )

    plot_histogram_with_statistics(
        v_dist, bins=41,
        title="Histogram of Vertical Distances for Matches",
        xlabel="Vertical Distance",
        ylabel="Frequency",
        filename=f'{save_dir}/{exp_name}_Histogram_vertical_dist_matches.png'
    )

    plot_histogram_with_statistics(
        h_dist, bins=41,
        title="Histogram of Horizontal Distances for Matches",
        xlabel="Horizontal Distance",
        ylabel="Frequency",
        filename=f'{save_dir}/{exp_name}_Histogram_horizontal_dist_matches.png'
    )

def remove_outliers_iqr(data):
    """
    Remove outliers using IQR method.

    Args:
        data (list): List of weighted distances.

    Returns:
        list: List of filtered distances without outliers.
    """
    sorted_data = sorted(data)
    q1 = np.percentile(sorted_data, 25)
    q3 = np.percentile(sorted_data, 75)
    iqr = q3 - q1
    upper_bound = q3 + 1.25 * iqr
    filtered_data = [x for x in data if x <= upper_bound]
    return filtered_data

def filter_metadata_with_iqr(match_metadata):
    """
    Filter out outliers from match_metadata using the IQR method.

    Args:
        match_metadata (list): List of match IDs and scores

    Returns:
        list: Filtered match metadata without outliers.
    """
    # Extract scores from match_metadata
    scores = [value['score'] for value in match_metadata]

    # Remove outliers using the IQR method
    filtered_scores = remove_outliers_iqr(scores)

    # Filter match_metadata based on the filtered scores
    filtered_metadata = [entry for entry in match_metadata if entry['score'] in filtered_scores]

    return filtered_metadata

def get_metadata(matches, offset):
    """
    Extract metadata from matches and produce a dictionary-based structure.
    
    Args:
        matches (list of tuples): List of centroid matches in the format (x1, y1, x2, y2).
        offset (int): Offset to adjust the full-scale centroid.
    
    Returns:
        list of dict: List of dictionaries containing metadata for each match.
    """
    match_metadata = []
    fourth_elements = set()

    for i, match in enumerate(matches):
        (x1, y1, x2, y2) = match['match']
        # Calculate Euclidean distance
        euclidean_distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        # Calculate vertical bias (weight horizontal distance more heavily)
        vertical_bias = abs(x2 - x1) * 2
        
        # Combine distance and bias into a weighted score
        weighted_score = euclidean_distance + vertical_bias
        
        # Determine match ID (Normal matches "N" or Reversed matches "R")
        match_id = f"N{i+1}" if x1 < x2 else f"R{i+1}"
        
        # Define centroids
        relative_centroid = (x1, y1, x2, y2)
        full_scale_centroid = (x1, y1, x2, y2 + offset)
        
        # Create dictionary for the match
        match_dict = {
            "match_id": match_id,
            "score": weighted_score,
            "relative_centroid": relative_centroid,
            "full_centroid": full_scale_centroid,
            "biased_score": match['biased_score'],
            "h_dist": (full_scale_centroid[0]-full_scale_centroid[2]),
            "v_dist": (full_scale_centroid[1]-full_scale_centroid[3]),
            'area_diff': match['area_diff'],
            'vol_diff': match['vol_diff'],
            'fmin_diff': match['fmin_diff'],
            'area1': match['area1'],
            'area2': match['area2'],
            'fmin1': match['fmin1'],
            'fmin2': match['fmin2'],
        }
        
        # Ensure no duplicates based on the full-scale centroid
        if full_scale_centroid not in fourth_elements:
            fourth_elements.add(full_scale_centroid)
            match_metadata.append(match_dict)

    return match_metadata

def compute_particle_metrics(contour):
    """
    Compute Feret min, Feret max, area, and estimated volume for a given particle contour.

    Args:
        contour (numpy.ndarray): Contour of the particle.

    Returns:
        tuple: (feret_min, feret_max, area, volume)
    """
    # Compute bounding rectangle
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # Compute Feret min and Feret max
    feret_min = np.min([np.linalg.norm(box[i] - box[(i + 1) % 4]) for i in range(4)])
    feret_max = np.max([np.linalg.norm(box[i] - box[(i + 1) % 4]) for i in range(4)])
    a = feret_min / 2
    b = feret_max / 2

    # Estimate volume using ellipsoidal approximation
    volume = (4 / 3) * np.pi * a * b

    area = cv2.contourArea(contour)

    # print(feret_min, feret_max, area, volume)

    return feret_min, feret_max, area, volume

def detect_centroids_with_metrics(image, threshold=150, edge_margin=10):
    """
    Detect centroids and compute metrics for all individual particles in an image.

    Args:
        image (numpy.ndarray): Input image (grayscale).
        threshold (int): Threshold value for binary segmentation.
        edge_margin (int): Margin from the edges to exclude particles.

    Returns:
        list: List of particle data [(x, y, feret_min, feret_max, area, volume), ...].
    """
    # Apply binary thresholding
    _, binary = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    inverted_image = cv2.bitwise_not(binary)

    # Find contours
    contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Extract centroids and metrics
    particle_data = []
    height, width = image.shape
    for contour in contours:
        # Compute metrics
        feret_min, feret_max, area, volume = compute_particle_metrics(contour)

        # Calculate moments for each contour
        M = cv2.moments(contour)
        if M["m00"] != 0:  # Avoid division by zero
            cx = int(M["m10"] / M["m00"])  # x-coordinate of centroid
            cy = int(M["m01"] / M["m00"])  # y-coordinate of centroid

            # Exclude particles near the edges
            if edge_margin <= cx <= width - edge_margin and edge_margin <= cy <= height - edge_margin:
                particle_data.append({'cx':cx, 'cy':cy, 
                                      'feret_min':feret_min, 'feret_max':feret_max , 
                                      'area': area, 'volume':volume})

    return particle_data

def filter_matches(matches, particle_data1, particle_data2, max_area_threshold=500):
    """
    Filter matches based on edge proximity, unique centroids, and area threshold.

    Args:
        matches (list): List of matched centroid pairs [(x1, y1, x2, y2), ...].
        particle_data1 (list): Particle data for image 1 [(x, y, feret_min, feret_max, area, volume), ...].
        particle_data2 (list): Particle data for image 2 [(x, y, feret_min, feret_max, area, volume), ...].
        max_area_threshold (float): Maximum area threshold for filtering.

    Returns:
        list: Filtered matches [(x1, y1, x2, y2), ...].
    """
    # Create dictionaries for quick lookup of metrics
    particle_dict1 = {(x, y): (feret_min, feret_max, area, volume) for x, y, feret_min, feret_max, area, volume in particle_data1}
    particle_dict2 = {(x, y): (feret_min, feret_max, area, volume) for x, y, feret_min, feret_max, area, volume in particle_data2}

    # Filter matches
    filtered_matches = []
    used_centroids1 = set()
    used_centroids2 = set()
    for match in matches:
        (x1, y1, x2, y2) = match['match']
        # Ensure unique centroids in both directions
        if (x1, y1) in used_centroids1 or (x2, y2) in used_centroids2:
            continue

        # Check area threshold
        area1 = particle_dict1.get((x1, y1), (None, None, float("inf"), None))[2]
        area2 = particle_dict2.get((x2, y2), (None, None, float("inf"), None))[2]
        if area1 <= max_area_threshold and area2 <= max_area_threshold:
            filtered_matches.append((x1, y1, x2, y2))
            used_centroids1.add((x1, y1))
            used_centroids2.add((x2, y2))

    return filtered_matches

def calculate_percentage_change(value1, value2):
    """
    Calculate percentage change between two values.

    Args:
        value1 (float): Initial value.
        value2 (float): Final value.

    Returns:
        float: Percentage change.
    """
    if value1 == 0:
        return 0
    return ((value2 - value1) / value1) * 100

def plot_area_histogram(particle_data, output_path='area_histogram.png'):
    """
    Plot a histogram of particle areas to help determine a proper threshold.

    Args:
        particle_data (list): List of particle data [(x, y, feret_min, feret_max, area, volume), ...].
        output_path (str): Path to save the histogram plot.
        bins (int): Number of bins for the histogram.

    Returns:
        None
    """
    # Extract areas from particle data
    areas = [area for x, y, feret_min, feret_max, area, volume in particle_data]
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(areas, bins=10, color='blue', alpha=0.7, edgecolor='black')
    plt.title("Histogram of Particle Areas", fontsize=16)
    plt.xlabel("Area", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Save the histogram
    plt.savefig(output_path)

def save_matches_to_csv(matches, particle_data1, particle_data2, output_path="matches.csv"):
    """
    Save matches and their metrics to a CSV file, including percentage changes.

    Args:
        matches (list): List of matched centroid pairs [(x1, y1, x2, y2), ...].
        particle_data1 (list): Particle data for image 1 [(x, y, feret_min, feret_max, area, volume), ...].
        particle_data2 (list): Particle data for image 2 [(x, y, feret_min, feret_max, area, volume), ...].
        output_path (str): Path to save the CSV file.

    Returns:
        None
    """
    # Create dictionaries for quick lookup of metrics
    particle_dict1 = {(x, y): (feret_min, feret_max, area, volume) for x, y, feret_min, feret_max, area, volume in particle_data1}
    particle_dict2 = {(x, y): (feret_min, feret_max, area, volume) for x, y, feret_min, feret_max, area, volume in particle_data2}

    # Prepare data for DataFrame
    data = []
    for match in matches:
        (x1, y1, x2, y2) = match['match']
        feret_min1, feret_max1, area1, volume1 = particle_dict1.get((x1, y1), (None, None, None, None))
        feret_min2, feret_max2, area2, volume2 = particle_dict2.get((x2, y2), (None, None, None, None))

        # Calculate percentage changes
        area_change = calculate_percentage_change(area1, area2)
        feret_min_change = calculate_percentage_change(feret_min1, feret_min2)
        feret_max_change = calculate_percentage_change(feret_max1, feret_max2)
        volume_change = calculate_percentage_change(volume1, volume2)

        data.append({
            "x1": x1, "y1": y1, "feret_min1": feret_min1, "feret_max1": feret_max1, "area1": area1, "volume1": volume1,
            "x2": x2, "y2": y2, "feret_min2": feret_min2, "feret_max2": feret_max2, "area2": area2, "volume2": volume2,
            "area_change (%)": area_change, "feret_min_change (%)": feret_min_change,
            "feret_max_change (%)": feret_max_change, "volume_change (%)": volume_change
        })

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

def save_tracked_particles_summary_to_csv(tracked_particles, output_path="tracked_particles_summary.csv"):
    """
    Save tracked particle data to a CSV file with metrics for up to three frames and summary columns.

    Args:
        tracked_particles (list): List of tracked particles with metrics for up to three frames.
        output_path (str): Path to save the CSV file.

    Returns:
        None
    """
    data = []
    for particle in tracked_particles:
        frame1 = particle["frame1"]
        frame2 = particle["frame2"]
        frame3 = particle["frame3"]

        # Extract metrics for each frame
        volume1 = frame1[5] if frame1 else None
        volume2 = frame2[5] if frame2 else None
        volume3 = frame3[5] if frame3 else None

        area1 = frame1[4] if frame1 else None
        area2 = frame2[4] if frame2 else None
        area3 = frame3[4] if frame3 else None

        feret_min1 = frame1[2] if frame1 else None
        feret_min2 = frame2[2] if frame2 else None
        feret_min3 = frame3[2] if frame3 else None

        feret_max1 = frame1[3] if frame1 else None
        feret_max2 = frame2[3] if frame2 else None
        feret_max3 = frame3[3] if frame3 else None

        # Compute summary metrics
        volumes = [v for v in [volume1, volume2, volume3] if v is not None]
        areas = [a for a in [area1, area2, area3] if a is not None]
        feret_mins = [fm for fm in [feret_min1, feret_min2, feret_min3] if fm is not None]
        feret_maxs = [fm for fm in [feret_max1, feret_max2, feret_max3] if fm is not None]

        avg_volume = np.mean(volumes) if volumes else None
        avg_area = np.mean(areas) if areas else None
        avg_feret_min = np.mean(feret_mins) if feret_mins else None
        avg_feret_max = np.mean(feret_maxs) if feret_maxs else None

        data.append({
            "volume-frame1": volume1,
            "volume-frame2": volume2,
            "volume-frame3": volume3,
            "area-frame1": area1,
            "area-frame2": area2,
            "area-frame3": area3,
            "feret_min-frame1": feret_min1,
            "feret_min-frame2": feret_min2,
            "feret_min-frame3": feret_min3,
            "feret_max-frame1": feret_max1,
            "feret_max-frame2": feret_max2,
            "feret_max-frame3": feret_max3,
            "avg_volume": avg_volume,
            "avg_area": avg_area,
            "avg_feret_min": avg_feret_min,
            "avg_feret_max": avg_feret_max
        })

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)

def loop_metrics_(input_dir, threshold=150, output_path="tracked_particles_summary.csv"):
    """
    Loop through all images in the directory and track particles across three consecutive frames.

    Args:
        input_dir (str): Directory containing the images.
        threshold (int): Threshold value for binary segmentation.
        max_distance (float): Maximum distance for matching centroids.
        output_path (str): Path to save the final CSV file.

    Returns:
        None
    """
    # Get all image paths sorted by name
    image_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".png")])

    # Load all frames
    frames = [cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) for image_path in image_paths]

    # Track particles across frames
    tracked_particles = track_particles_limited(frames, threshold=threshold)

    # Save tracked particle data to CSV
    save_tracked_particles_summary_to_csv(tracked_particles, output_path=output_path)

    print(f"Tracked particle summary saved to {output_path}")

def example():
    f1 = '/projects/OLIVINE/data/input_device/Oryx/oryx/olivine/olivine_60_144fps/frame_0006.png'
    f2 = '/projects/OLIVINE/data/input_device/Oryx/oryx/olivine/olivine_60_144fps/frame_0007.png'
    f3 = '/projects/OLIVINE/data/input_device/Oryx/oryx/olivine/olivine_60_144fps/frame_0008.png'

    image1_path = f2
    image2_path = f3

    # Load images
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Perform template matching to extract sub-images
    top_third_template, matched_region, max_loc = template_matching(image1, image2)

    # Detect centroids in both sub-images using moments and visualize binary images
    centroids1, color1 = detect_centroids_using_moments(top_third_template, color=(255, 0, 0), threshold=80, visualize_binary=True)
    centroids2, color2 = detect_centroids_using_moments(matched_region, color=(0, 255, 0), threshold=80, visualize_binary=True)
    # Perform template matching to extract sub-images and match location

    # Combine the top third of the first image onto the full second image
    combined_image = color1 + color2
    img = Image.fromarray(combined_image)
    img.save('combined_image.png')

    # Visualize detected centroids on each sub-image
    visualize_centroids(cv2.cvtColor(top_third_template, cv2.COLOR_GRAY2RGB), centroids1, title="Detected Centroids in Image 1")
    visualize_centroids(cv2.cvtColor(matched_region, cv2.COLOR_GRAY2RGB), centroids2, title="Detected Centroids in Image 2")
    
    # Perform bidirectional matching with vertical priority and natural unmatched detection
    matches = combine_bidirectional_matches_with_vertical_priority(centroids1, centroids2)
    print(len(matches))
    analyze_and_plot_weighted_distances(matches)

    # Extract IDs and vertical distances
    match_metadata = get_metadata(matches)

    # Visualize tracking with color scale
    visualize_tracking_with_color_scale(
        cv2.cvtColor(top_third_template, cv2.COLOR_GRAY2RGB),
        cv2.cvtColor(matched_region, cv2.COLOR_GRAY2RGB),
        match_metadata
    )

    # Filter metadata using IQR
    filtered_metadata = filter_metadata_with_iqr(match_metadata)
    # Filter matches based on filtered metadata
    max_score = max(match_metadata, key=lambda x: x[1])[1]
    print(max_score)

    # Re-visualize tracking with filtered metadata
    visualize_tracking_with_color_scale(
        cv2.cvtColor(top_third_template, cv2.COLOR_GRAY2RGB),
        cv2.cvtColor(matched_region, cv2.COLOR_GRAY2RGB),
        filtered_metadata,
        max_score=max_score
    )

def metrics_example():

    f1 = '/projects/OLIVINE/data/input_device/Oryx/oryx/olivine/olivine_60_144fps/frame_0006.png'
    f2 = '/projects/OLIVINE/data/input_device/Oryx/oryx/olivine/olivine_60_144fps/frame_0007.png'
    f3 = '/projects/OLIVINE/data/input_device/Oryx/oryx/olivine/olivine_60_144fps/frame_0008.png'

    image1_path = f2
    image2_path = f3

    # Load images
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Perform template matching to extract sub-images
    top_third_template, matched_region, max_loc = template_matching(image1, image2)

    # Detect centroids and metrics
    particle_data1 = detect_centroids_with_metrics(top_third_template, threshold=80, edge_margin=10)
    particle_data2 = detect_centroids_with_metrics(matched_region, threshold=80, edge_margin=10)
    
    # Perform bidirectional matching
    centroids1 = [(x, y) for x, y, feret_min, feret_max, area, volume in particle_data1]
    centroids2 = [(x, y) for x, y, feret_min, feret_max, area, volume in particle_data2]
    matches = combine_bidirectional_matches_with_vertical_priority(centroids1, centroids2)
    particle_all = particle_data1 + particle_data2
    plot_area_histogram(particle_all)

    # Filter matches
    filtered_matches = filter_matches(matches, particle_data1, particle_data2, max_area_threshold=10000)
    # Save matches to CSV
    save_matches_to_csv(filtered_matches, particle_data1, particle_data2, output_path="filtered_matches.csv")

def visualize_tracking_three_frames_side_by_side(image1, image2, image3, match_metadata12, match_metadata23, output_path="tracking_visualization_three_frames.png"):
    """
    Visualize particle tracking across three frames, stacking the images side by side, 
    and use a color scale (Jet colormap) to represent weighted distances of the match lines.

    Args:
        image1 (numpy.ndarray): First image (RGB or grayscale).
        image2 (numpy.ndarray): Second image (RGB or grayscale).
        image3 (numpy.ndarray): Third image (RGB or grayscale).
        match_metadata12 (list): Match metadata between frame 1 and frame 2 [(match_id, distance, template_scale, full_scale_centroids)].
        match_metadata23 (list): Match metadata between frame 2 and frame 3 [(match_id, distance, template_scale, full_scale_centroids)].
        output_path (str): Path to save the visualization image.

    Returns:
        None
    """
    # Convert images to RGB if they are grayscale
    if len(image1.shape) == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
    if len(image2.shape) == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)
    if len(image3.shape) == 2:
        image3 = cv2.cvtColor(image3, cv2.COLOR_GRAY2RGB)

    # Stack the images horizontally for visualization
    combined_image = np.hstack((image1, image2, image3))

    # Extract weighted scores for normalization
    weighted_scores12 = [value['score'] for value in match_metadata12]
    weighted_scores23 = [value['score'] for value in match_metadata23]
    max_distance = max(weighted_scores12 + weighted_scores23) if (weighted_scores12 + weighted_scores23) else 1  # Avoid division by zero

    # Normalize distances to [0, 1] range
    normalized_distances12 = [score / max_distance for score in weighted_scores12]
    normalized_distances23 = [score / max_distance for score in weighted_scores23]

    # Get Jet colormap
    colormap = plt.colormaps["jet"]

    # Draw tracking lines and labels for frame 1 → frame 2
    for i, value in enumerate(match_metadata12):
        match_id = value['match_id']
        (x1, y1, x2, y2) = value['full_centroid']
        x2_adjusted = x2 + image1.shape[1]  # Adjust x-coordinate for the second image

        # Normalize the weighted score for color mapping
        normalized_distance = normalized_distances12[i]

        # Get color from Jet colormap (convert to BGR format for OpenCV)
        color = colormap(normalized_distance)[:3]  # RGB values
        color_bgr = tuple([int(255 * c) for c in reversed(color)])  # Convert to BGR for OpenCV

        # Draw a line connecting the centroids
        cv2.line(combined_image, (x1, y1), (x2_adjusted, y2), color_bgr, 2)
        cv2.circle(combined_image, (x1, y1), 5, (0, 255, 0), -1)  # Green circle for the first image
        cv2.circle(combined_image, (x2_adjusted, y2), 5, (0, 0, 255), -1)  # Blue circle for the second image

        # Add text label for the match ID
        label_position = ((x1 + x2_adjusted) // 2, (y1 + y2) // 2)
        cv2.putText(combined_image, match_id, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw tracking lines and labels for frame 2 → frame 3
    for i, value in enumerate(match_metadata23):
        match_id = value['match_id']
        (x2, y2, x3, y3) = value['full_centroid']
        x2_adjusted = x2 + image1.shape[1]  # Adjust x-coordinate for the second image
        x3_adjusted = x3 + image1.shape[1] + image2.shape[1]  # Adjust x-coordinate for the third image

        # Normalize the weighted score for color mapping
        normalized_distance = normalized_distances23[i]

        # Get color from Jet colormap (convert to BGR format for OpenCV)
        color = colormap(normalized_distance)[:3]  # RGB values
        color_bgr = tuple([int(255 * c) for c in reversed(color)])  # Convert to BGR for OpenCV

        # Draw a line connecting the centroids
        cv2.line(combined_image, (x2_adjusted, y2), (x3_adjusted, y3), color_bgr, 2)
        cv2.circle(combined_image, (x2_adjusted, y2), 5, (0, 255, 0), -1)  # Green circle for the second image
        cv2.circle(combined_image, (x3_adjusted, y3), 5, (0, 0, 255), -1)  # Blue circle for the third image

        # Add text label for the match ID
        label_position = ((x2_adjusted + x3_adjusted) // 2, (y2 + y3) // 2)
        cv2.putText(combined_image, match_id, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # # Display the result using matplotlib
    plt.figure(figsize=(20, 8), dpi=300)
    plt.imshow(combined_image)
    plt.title("Particle Tracking Across Three Frames (Side by Side)")
    plt.axis("off")
    plt.savefig(output_path)
    print(f"Tracking visualization saved to {output_path}")

def image_pair_metrics(image1_path, image2_path):
    """
    Compute metrics for a pair of images and return filtered matches.

    Args:
        image1_path (str): Path to the first image.
        image2_path (str): Path to the second image.

    Returns:
        list: Filtered matches [(x1, y1, x2, y2), ...].
    """
    # Load images
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # Perform template matching to extract sub-images
    top_third_template, matched_region, max_loc = template_matching(image1, image2)

    # Detect centroids and metrics
    particle_data1 = detect_centroids_with_metrics(top_third_template, threshold=80, edge_margin=10)
    particle_data2 = detect_centroids_with_metrics(matched_region, threshold=80, edge_margin=10)
    
    # Perform bidirectional matching
    centroids1 = [(x, y) for x, y, feret_min, feret_max, area, volume in particle_data1]
    centroids2 = [(x, y) for x, y, feret_min, feret_max, area, volume in particle_data2]
    matches = combine_bidirectional_matches_with_vertical_priority(centroids1, centroids2)

    # Filter matches
    filtered_matches = filter_matches(matches, particle_data1, particle_data2, max_area_threshold=10000)

    return filtered_matches, particle_data1, particle_data2

def loop_metrics(input_dir, depth=3, output_path="filtered_matches.csv"):
    """
    Loop through all images in the directory and compute metrics for each pair of images.

    Args:
        input_dir (str): Directory containing the images.
        depth (int): Number of subsequent frames to compare with each image.
        output_path (str): Path to save the final CSV file.

    Returns:
        None
    """
    # Get all image paths sorted by name
    image_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".png")])

    # Initialize a list to store data for the DataFrame
    all_data = []

    # Loop through each image
    for i in tqdm(range(len(image_paths) - depth), desc="Processing image pairs"):
        # Get the current image path
        image1_path = image_paths[i]

        # Compare with subsequent images up to the specified depth
        for d in range(1, depth + 1):
            if i + d < len(image_paths):
                image2_path = image_paths[i + d]

                # Compute metrics for the image pair
                filtered_matches, particle_data1, particle_data2 = image_pair_metrics(image1_path, image2_path)

                # Create dictionaries for quick lookup of metrics
                particle_dict1 = {(x, y): (feret_min, feret_max, area, volume) for x, y, feret_min, feret_max, area, volume in particle_data1}
                particle_dict2 = {(x, y): (feret_min, feret_max, area, volume) for x, y, feret_min, feret_max, area, volume in particle_data2}

                # Add data for each match to the list
                for x1, y1, x2, y2 in filtered_matches:
                    feret_min1, feret_max1, area1, volume1 = particle_dict1.get((x1, y1), (None, None, None, None))
                    feret_min2, feret_max2, area2, volume2 = particle_dict2.get((x2, y2), (None, None, None, None))

                    # Calculate percentage changes
                    area_change = calculate_percentage_change(area1, area2)
                    feret_min_change = calculate_percentage_change(feret_min1, feret_min2)
                    feret_max_change = calculate_percentage_change(feret_max1, feret_max2)
                    volume_change = calculate_percentage_change(volume1, volume2)

                    # Append data to the list
                    all_data.append({
                        "frame1": os.path.basename(image1_path),
                        "frame2": os.path.basename(image2_path),
                        "x1": x1, "y1": y1, "feret_min1": feret_min1, "feret_max1": feret_max1, "area1": area1, "volume1": volume1,
                        "x2": x2, "y2": y2, "feret_min2": feret_min2, "feret_max2": feret_max2, "area2": area2, "volume2": volume2,
                        "area_change (%)": area_change, "feret_min_change (%)": feret_min_change,
                        "feret_max_change (%)": feret_max_change, "volume_change (%)": volume_change
                    })

    # Create a DataFrame from the collected data
    df = pd.DataFrame(all_data)

    # Save the DataFrame to a CSV file
    df.to_csv(output_path, index=False)

    print(f"Metrics saved to {output_path}")

def track_particles_across_frames(frames, threshold=150, overlap_ratio=0.75):
    """
    Track particles across all frames by daisy chaining matches.

    Args:
        frames (list): List of grayscale images.
        threshold (int): Threshold value for binary segmentation.
        overlap_ratio (float): Ratio of the second image to consider for matching.

    Returns:
        pd.DataFrame: DataFrame of tracked particles across frames.
    """
    tracked_particles = []

    # Initialize particle data for the first frame
    particle_data_prev = detect_centroids_with_metrics(frames[0], threshold=threshold)

    # Loop through all frames
    for frame_idx in tqdm(range(1, len(frames)), desc="Tracking particles across frames"):
        # Perform template matching between consecutive frames
        top_third_template, matched_region, _ = template_matching(frames[frame_idx - 1], frames[frame_idx], overlap_ratio=overlap_ratio)

        # Detect centroids and metrics in both sub-images
        particle_data_curr = detect_centroids_with_metrics(matched_region, threshold=threshold)

        # Extract centroids for matching
        centroids_prev = [(x, y) for x, y, _, _, _, _ in particle_data_prev]
        centroids_curr = [(x, y) for x, y, _, _, _, _ in particle_data_curr]

        # Perform bidirectional matching
        matches = combine_bidirectional_matches_with_vertical_priority(centroids_prev, centroids_curr)

        # Filter matches based on metrics
        filtered_matches = filter_matches(matches, particle_data_prev, particle_data_curr, max_area_threshold=10000)

        # Update tracked particles
        for x1, y1, x2, y2 in filtered_matches:
            tracked_particles.append({
                "frame_idx_prev": frame_idx - 1,
                "frame_idx_curr": frame_idx,
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2
            })

        # Update particle data for the next iteration
        particle_data_prev = particle_data_curr

    # Convert tracked particles to a DataFrame
    df = pd.DataFrame(tracked_particles)
    return df

def save_tracked_particles_to_csv(tracked_particles_df, output_path="tracked_particles.csv"):

    """
    Save tracked particle data to a CSV file.

    Args:
        tracked_particles_df (pd.DataFrame): DataFrame of tracked particles across frames.
        output_path (str): Path to save the CSV file.

    Returns:
        None
    """
    tracked_particles_df.to_csv(output_path, index=False)
    print(f"Tracked particle data saved to {output_path}")

def loop_metrics_across_frames(input_dir, threshold=150, overlap_ratio=0.75, output_path="tracked_particles.csv"):
    """
    Loop through all images in the directory and track particles across frames.

    Args:
        input_dir (str): Directory containing the images.
        threshold (int): Threshold value for binary segmentation.
        overlap_ratio (float): Ratio of the second image to consider for matching.
        output_path (str): Path to save the final CSV file.

    Returns:
        None
    """
    # Get all image paths sorted by name
    image_paths = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".png")])

    # Load all frames
    frames = [cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) for image_path in image_paths]

    # Track particles across frames
    tracked_particles_df = track_particles_across_frames(frames, threshold=threshold, overlap_ratio=overlap_ratio)

    # Save tracked particle data to CSV
    save_tracked_particles_to_csv(tracked_particles_df, output_path=output_path)