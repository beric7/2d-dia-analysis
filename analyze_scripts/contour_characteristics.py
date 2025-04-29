import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from utils.dce import dce_area
import feret
import time
from utils.smallestenclosingcircle import smallestenclosingcircle as sec
from utils.load_json import load_json_data 
import json

# Calculate metrics
from grain_analysis_functions import compute_sphericity, compute_sphericity_all, compute_volume, ellipsoidal_volume

# Get contours of the detected shapes, draw edges. 
def get_contours(image_result):
    # Find contours in the binary image
    contours, _ = cv2.findContours(image_result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Optional trim contour method using DCE
def trim_contour(contour_path, dce_pass=0, area=True):
    if len(contour_path) > 2:
        trimmed_contour = dce_area(dce_pass, contour_path)
    return trimmed_contour

# Fill contour for feret calculations
def fill_contour(inverted_image, contour):
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

# For each contour, get the area and calculate the moment/centroid of the contour. 
# Use the centroids to inform SAM.
def get_moments(image_result, image_color, save_dir, img_name):
    # Find contours in the binary image
    contours = get_contours(image_result)
    contour_color = (0, 255, 0)
    contour_thickness = 2
    cv2.drawContours(image_color, contours, -1, contour_color, contour_thickness)

    # Save the image with contours
    output_image_path = f'{save_dir}/contour_{img_name}'
    cv2.imwrite(output_image_path, image_color)
    
    centroids = []
    # print(len(contours))
    for cont in contours:
        if cv2.contourArea(cont) > 25:            
            contour = cont
            # Calculate moments for each contour
            M = cv2.moments(contour)
            
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroids.append((cx, cy))
            else:
                # Handle the case where the contour area is zero
                continue

    print(f'Number of centroids: {len(centroids)}')
    return centroids

# Get positional information for SAM
def get_positions(imageFilepath, save_dir = '/projects/OLIVINE/data/saved_otsu_contours_img_11am/'):
    # Read the image in a grayscale mode
    image = cv2.imread(imageFilepath, 0)
    file_name = os.path.basename(imageFilepath)
    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # image_grey = image.copy()

    #otsu's thresholding
    # ============================================================
    otsu_threshold, image_result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # print(f'threshold is: {otsu_threshold}')
    inverted_image = cv2.bitwise_not(image_result)
    if otsu_threshold < 80:
        kernel = np.ones((20,20),np.uint8)
        eroded_img = cv2.erode(inverted_image, kernel, iterations = 1)
        cv2.imwrite(f'{save_dir}/threshold_bin_{file_name}', inverted_image)
        centroids = get_moments(eroded_img, image_color, save_dir, file_name)
        # print(len(centroids))
        return centroids
    else:
        print(f'Image {file_name} is blank')
        cv2.imwrite(f'{save_dir}/threshold_bin_{file_name}', inverted_image)
        output_image_path = f'{save_dir}/contour_{file_name}'
        cv2.imwrite(output_image_path, image_color)

# Organize contour data and apply calculations to contours
def collect_contours(inverted_image, file_name, save_grain_img_dir, PIXEL_TO_MM) -> pd.DataFrame:
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
            # cv2.imwrite(f'{save_grain_img_dir}/{name}_contour_{i}.jpg', cropped)
            # print(trimmed_contour)
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
        i+=1
    # Perform element-wise multiplication using list comprehensions
    feret_max_mm = [value * PIXEL_TO_MM for value in feretMax_list]
    feret_min_mm = [value * PIXEL_TO_MM for value in feretMin_list]
    data = {'file_name': fileName, 'width': width, 'height': height, 'contour': contour_list, 
            'touch_edge': edge_list, 'area': area_list, 
            'sphericity': sphericity_list, 'feret_max': feretMax_list, 
            'feret_max_mm': feret_max_mm, 'feret_min_mm': feret_min_mm,
            'feret_min': feretMin_list, 'feret_max_angle': feretMaxAngle_list, 
            'feret_min_angle': feretMinAngle_list, 'centroid': centroid_list, 'volume' :volume_list, 'ellip_volume': ellip_vol_list}
    df = pd.DataFrame(data) 
    return df

# apply thresholding and call 'collect contours'
def get_contour_df(imageFilepath, save_grain_img_dir, PIXEL_TO_MM):
    # Read the image in a grayscale mode
    image = cv2.imread(imageFilepath, 0)
    file_name = os.path.basename(imageFilepath)

    # Apply adaptive thresholding
    thersh, thresholded_image = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY) # - TELEDYNE (20), Oryx (80)
    # otsu_threshold, thresholded_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # print(f'threshold is: {otsu_threshold}')
    inverted_image = cv2.bitwise_not(thresholded_image)
    cv2.imwrite(f'{save_grain_img_dir}/threshold_bin_{file_name}', inverted_image)
    contour_df = collect_contours(inverted_image, file_name, save_grain_img_dir, PIXEL_TO_MM)
    return contour_df
