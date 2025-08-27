# Copyright 2025 The MITRE Corporation

import cv2
import numpy as np
import os
import pandas as pd
from utils.dce import dce_area
import feret

# Calculate metrics
from grain_analysis_functions import compute_sphericity_all, compute_volume, ellipsoidal_volume

def get_contours(image_result: np.ndarray) -> list:
    """
    Find contours in a binary image.

    Args:
        image_result (np.ndarray): Binary image.

    Returns:
        -> list: List of contours found in the image.
    """
    contours, _ = cv2.findContours(image_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def trim_contour(contour_path: np.ndarray, dce_pass: int = 0, area: bool = True) -> np.ndarray:
    """
    Optionally trim a contour using DCE (Discrete Curve Evolution).

    Args:
        contour_path (np.ndarray): The contour to trim.
        dce_pass (int, optional): Number of DCE passes. Default is 0.
        area (bool, optional): Whether to use area-based DCE. Default is True.

    Returns:
        -> np.ndarray: Trimmed contour.
    """
    if len(contour_path) > 2:
        trimmed_contour = dce_area(dce_pass, contour_path)
        return trimmed_contour
    return contour_path

def fill_contour(inverted_image: np.ndarray, contour: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Fill a contour for Feret calculations and check if it touches the image edge.

    Args:
        inverted_image (np.ndarray): Inverted binary image.
        contour (np.ndarray): Contour to fill.

    Returns:
        -> tuple[np.ndarray, bool]: Cropped mask of the filled contour, and whether it touches the edge.
    """
    height, width = inverted_image.shape[:2]
    mask = np.zeros_like(inverted_image)
    cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

    # Find bounding box of contour and crop the region
    x, y, w, h = cv2.boundingRect(contour)

    # Check if the contour touches the edge of the original image
    touches_edge = x == 0 or y == 0 or x + w == width or y + h == height

    # Add padding
    padding = 10
    padded_x = max(x - padding, 0)
    padded_y = max(y - padding, 0)
    padded_w = min(w + 2 * padding, width - padded_x)
    padded_h = min(h + 2 * padding, height - padded_y)

    # Crop the padded region
    cropped = mask[padded_y:padded_y + padded_h, padded_x:padded_x + padded_w]

    return cropped, touches_edge

def get_moments(
    image_result: np.ndarray,
    image_color: np.ndarray,
    save_dir: str,
    img_name: str
) -> list[tuple[int, int]]:
    """
    For each contour, get the area and calculate the centroid (moment).
    Draw contours and save the result image.

    Args:
        image_result (np.ndarray): Binary image for contour detection.
        image_color (np.ndarray): Color image for drawing contours.
        save_dir (str): Directory to save output images.
        img_name (str): Name of the image file.

    Returns:
        -> list[tuple[int, int]]: List of centroid coordinates (cx, cy).
    """
    contours = get_contours(image_result)
    contour_color = (0, 255, 0)
    contour_thickness = 2
    cv2.drawContours(image_color, contours, -1, contour_color, contour_thickness)

    # Save the image with contours
    output_image_path = f'{save_dir}/contour_{img_name}'
    cv2.imwrite(output_image_path, image_color)
    
    centroids = []
    for cont in contours:
        if cv2.contourArea(cont) > 25:            
            M = cv2.moments(cont)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroids.append((cx, cy))
            else:
                continue

    print(f'Number of centroids: {len(centroids)}')
    return centroids

def get_positions(
    imageFilepath: str,
    save_dir: str = '/projects/OLIVINE/data/saved_otsu_contours_img_11am/'
) -> list[tuple[int, int]]:
    """
    Get centroid positions for SAM by thresholding and finding contours.

    Args:
        imageFilepath (str): Path to the input image.
        save_dir (str, optional): Directory to save thresholded and contour images.

    Returns:
        -> list[tuple[int, int]]: List of centroid coordinates.
    """
    image = cv2.imread(imageFilepath, 0)
    file_name = os.path.basename(imageFilepath)
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Otsu's thresholding
    otsu_threshold, image_result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted_image = cv2.bitwise_not(image_result)
    if otsu_threshold < 80:
        kernel = np.ones((20,20),np.uint8)
        eroded_img = cv2.erode(inverted_image, kernel, iterations=1)
        cv2.imwrite(f'{save_dir}/threshold_bin_{file_name}', inverted_image)
        centroids = get_moments(eroded_img, image_color, save_dir, file_name)
        return centroids
    else:
        print(f'Image {file_name} is blank')
        cv2.imwrite(f'{save_dir}/threshold_bin_{file_name}', inverted_image)
        output_image_path = f'{save_dir}/contour_{file_name}'
        cv2.imwrite(output_image_path, image_color)
        return []

def collect_contours(
    inverted_image: np.ndarray,
    file_name: str,
    save_grain_img_dir: str,
    PIXEL_TO_MM: float
) -> pd.DataFrame:
    """
    Organize contour data and apply calculations to contours.

    Args:
        inverted_image (np.ndarray): Inverted binary image.
        file_name (str): Name of the image file.
        save_grain_img_dir (str): Directory to save grain images.
        PIXEL_TO_MM (float): Conversion factor from pixels to millimeters.

    Returns:
        -> pd.DataFrame: DataFrame containing contour and particle metrics.
    """
    contours = get_contours(inverted_image)
    contour_list = []
    area_list = []
    centroid_list = []
    edge_list = []
    fileName = []
    width = []
    height = []
    sphericity_list = []
    feretMin_list = []
    feretMax_list = []
    feretMinAngle_list = []
    feretMaxAngle_list = []
    volume_list = []
    ellip_vol_list = []

    i = 0
    for cont in contours:
        area = cv2.contourArea(cont)
        if area > 25: 
            area_list.append(area)
            trimmed_contour = trim_contour(cont)
            h, w = inverted_image.shape[:2]
            cropped, touches_edge = fill_contour(inverted_image, trimmed_contour)
            feret_result = feret.calc(cropped)
            name = file_name.split('.jpg')[0]
            sphericity, circle = compute_sphericity_all(cropped, area)
            sphericity_list.append(sphericity)
            feretMin_list.append(feret_result.minf)
            feretMax_list.append(feret_result.maxf)
            feretMinAngle_list.append(feret_result.minf_angle)
            feretMaxAngle_list.append(feret_result.maxf_angle)
            volume_list.append(compute_volume(feret_result, PIXEL_TO_MM))
            ellip_vol_list.append(ellipsoidal_volume(feret_result.minf, feret_result.maxf, PIXEL_TO_MM))
            edge_list.append(touches_edge)
            contour_list.append(trimmed_contour)       
            # Calculate moments for each contour
            M = cv2.moments(trimmed_contour)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroid_list.append((cx, cy))
            width.append(w)
            height.append(h)
            fileName.append(file_name)
        i += 1

    # Convert Feret diameters to millimeters
    feret_max_mm = [value * PIXEL_TO_MM for value in feretMax_list]
    feret_min_mm = [value * PIXEL_TO_MM for value in feretMin_list]
    data = {
        'file_name': fileName,
        'width': width,
        'height': height,
        'contour': contour_list, 
        'touch_edge': edge_list,
        'area': area_list, 
        'sphericity': sphericity_list,
        'feret_max': feretMax_list, 
        'feret_max_mm': feret_max_mm,
        'feret_min_mm': feret_min_mm,
        'feret_min': feretMin_list,
        'feret_max_angle': feretMaxAngle_list, 
        'feret_min_angle': feretMinAngle_list,
        'centroid': centroid_list,
        'volume': volume_list,
        'ellip_volume': ellip_vol_list
    }
    df = pd.DataFrame(data) 
    return df

def get_contour_df(
    imageFilepath: str,
    save_grain_img_dir: str,
    PIXEL_TO_MM: float
) -> pd.DataFrame:
    """
    Apply thresholding and collect contour data from an image.

    Args:
        imageFilepath (str): Path to the input image.
        save_grain_img_dir (str): Directory to save grain images.
        PIXEL_TO_MM (float): Conversion factor from pixels to millimeters.

    Returns:
        -> pd.DataFrame: DataFrame containing contour and particle metrics.
    """
    image = cv2.imread(imageFilepath, 0)
    file_name = os.path.basename(imageFilepath)

    # Apply adaptive thresholding
    thersh, thresholded_image = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY) # - TELEDYNE (20), Oryx (80)
    inverted_image = cv2.bitwise_not(thresholded_image)
    cv2.imwrite(f'{save_grain_img_dir}/threshold_bin_{file_name}', inverted_image)
    contour_df = collect_contours(inverted_image, file_name, save_grain_img_dir, PIXEL_TO_MM)
    return contour_df
