import os
import json
import glob
import numpy as np
from grain_analysis_functions import compute_sphericity_all, compute_volume_simple, ellipsoidal_volume, compute_feret_diameters
import feret
import cv2

from concurrent.futures import ProcessPoolExecutor


def combine_json_files(input_dir, output_file):
    """
    Combine all JSON files in a directory into a single JSON file.

    Parameters:
    - input_dir: str, path to the directory containing JSON files.
    - output_file: str, path to the combined JSON file to be saved.
    """
    combined_data = []

    # Get all JSON files in the input directory
    json_files = glob.glob(os.path.join(input_dir, "*.json"))

    for json_file in json_files:
        # Load the JSON data
        with open(json_file, "r") as f:
            data = json.load(f)

        # Append the data to the combined list
        combined_data.extend(data)

    # Save the combined data to the output file
    with open(output_file, "w") as f:
        json.dump(combined_data, f)

    print(f"Combined JSON file saved to: {output_file}")

def reshape_contour(contours):
    """
    Select the largest contour from a list of contours and reshape it into a 2D array of shape (n, 2).
    """
    if not contours or len(contours) == 0:
        raise ValueError("Contours list is empty or None.")

    # Find the largest contour based on its length
    largest_contour = max(contours, key=lambda c: len(c))

    # Reshape the largest contour into a 2D array of shape (n, 2)
    return np.array(largest_contour, dtype=np.int32).reshape(-1, 2)

def calculate_attributes(contour, area, img_width, img_height, pixel_2_mm):
    """
    Calculate attributes like sphericity, volume, and ellipsoidal volume for a given contour.
    """
    # Compute sphericity
    sphericity, circle = compute_sphericity_all(contour, area)

    # Compute Feret diameters
    # Compute Feret diameters using OpenCV
    feret_min, feret_max = compute_feret_diameters(contour)
    # print(f"Feret Min: {feret_min}, Feret Max: {feret_max}")

    # Compute volumes
    volume = compute_volume_simple(feret_min, feret_max, pixel_2_mm)
    ellip_volume = ellipsoidal_volume(feret_min, feret_max, pixel_2_mm)

    x, y, w, h = cv2.boundingRect(contour)

    # Check if the contour touches the edge of the original image
    touches_edge = x == 0 or y == 0 or x + w == img_width or y + h == img_height

    return {
        "touch_edge": touches_edge,
        "sphericity": sphericity,
        "area": area * pixel_2_mm ** 2, 
        "circle_radius": circle[2],
        "feret_min": feret_min,
        "feret_min_mm": feret_min * pixel_2_mm,
        "feret_max": feret_max,
        "feret_max_mm": feret_max * pixel_2_mm,
        "volume": volume,
        "ellip_volume": ellip_volume,
    }

def process_single_file(json_file, output_dir, pixel_2_mm):
    """
    Process a single JSON file, calculate attributes, and save updated JSON file to the output directory.
    """
    
    # Load the JSON data
    with open(json_file, "r") as f:
        data = json.load(f)

    updated_data = []
    for prediction in data:
        try:
            if "segmentation" in prediction:
                contours = reshape_contour(prediction["segmentation"])
                area = cv2.contourArea(np.array(contours, dtype=np.int32))  # Calculate area from contour
                img_width = prediction['width']
                img_height = prediction['height']
                # Perform attribute calculations
                attributes = calculate_attributes(contours, area, img_width, img_height, pixel_2_mm)

                # Update the prediction with calculated attributes
                prediction.update(attributes)

            updated_data.append(prediction)
        except:
            print(f"prediction did not compute: {prediction}")

    # Save the updated JSON data to the output directory
    output_file = os.path.join(output_dir, os.path.basename(json_file))
    with open(output_file, "w") as f:
        json.dump(updated_data, f)

def process_json_files(input_dir, output_dir, pixel_2_mm):
    """
    Process JSON files in a directory, calculate attributes, and save updated JSON files to a new directory.

    Parameters:
    - input_dir: str, path to the directory containing JSON files.
    - output_dir: str, path to the directory where updated JSON files will be saved.
    - pixel_2_mm: conversion factor from pixels to millimeters.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all JSON files in the input directory
    json_files = glob.glob(os.path.join(input_dir, "*.json"))

    # Use parallel processing to speed up file processing
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_single_file, json_file, output_dir, pixel_2_mm)
            for json_file in json_files
        ]
        for future in futures:
            future.result()  # Wait for all tasks to complete

def process_annotations(data, pixel_2_mm):
    """
    Process annotations to convert them into the desired format.
    """
    processed_data = []

    for annotation in data["annotations"]:
        if "a_segm" in annotation:
            # Reshape the contour
            contour = reshape_contour(annotation["a_segm"])

            # Calculate area
            area = cv2.contourArea(contour)

            # Calculate bounding box
            bbox = annotation["a_bbox"]

            # Calculate attributes
            attributes = calculate_attributes(contour, area, pixel_2_mm)

            # Prepare the processed annotation
            processed_annotation = {
                "bbox": bbox,
                "category_id": annotation["category_id"],
                "segmentation": annotation["a_segm"],
                **attributes
            }

            processed_data.append(processed_annotation)

    return processed_data