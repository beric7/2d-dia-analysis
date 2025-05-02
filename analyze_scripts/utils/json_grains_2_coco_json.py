import json
import os
from datetime import datetime

import json
import os
from datetime import datetime

import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import seaborn as sns
from scipy.ndimage import gaussian_filter  # For smoothing the heatmap

# Create histograms and subplots
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

# convert json data to COCO formatted data
def convert_to_coco(json_files, image_dir):
    coco_format = {
        'info': {
            'date_created': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'description': 'Sand Data in COCO format for BCNet'
        },
        'categories': [{'id': 1, 'name': 'particle'}],
        'licenses': None,
        'images': [],
        'annotations': []
    }

    annotation_id = 1

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        print(data['file_name'])
        file_name = data['file_name']['0']
        width = data['width']['0']
        height = data['height']['0']
        contours = data['contour']
        areas = data['area']
        feret_maxes = data['feret_max']
        feret_mins = data['feret_min']
        sphericities = data['sphericity']
        touch_edges = data['touch_edge']
        volumes = data['volume']
        ellip_volumes = data['ellip_volume']
        feret_min_mms = data['feret_min_mm']
        feret_max_mms = data['feret_max_mm']

        print(image_dir)
        print(file_name)
        # Add image information once per file
        image_info = {
            'id': file_name,
            'width': width,
            'height': height,
            'file_name': os.path.join(image_dir, file_name)
        }
        coco_format['images'].append(image_info)

        # Add annotations for each contour
        for contour_id, contour_points in contours.items():
            # Flatten the contour points for segmentation
            segmentation = [coord for point in contour_points for coord in point[0]]

            # Calculate bounding box
            x_coords = segmentation[0::2]  # Extract x coordinates
            y_coords = segmentation[1::2]  # Extract y coordinates
            x_min, y_min = min(x_coords), min(y_coords)
            x_max, y_max = max(x_coords), max(y_coords)
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

            # Get the area from the area list
            area = areas[contour_id]
            feret_max = feret_maxes[contour_id]
            feret_min = feret_mins[contour_id]
            sphericity = sphericities[contour_id]
            touch_edge = touch_edges[contour_id]
            volume = volumes[contour_id]
            ellip_volume = ellip_volumes[contour_id]
            feret_max_mm = feret_max_mms[contour_id]
            feret_min_mm = feret_min_mms[contour_id]

            annotation = {
                'id': annotation_id,
                'image_id': file_name,
                'bbox': bbox,
                'feret_max': feret_max, 
                'feret_max_mm': feret_max_mm,
                'feret_min': feret_min,
                'feret_min_mm': feret_min_mm,
                'sphericity': sphericity, 
                'touch_edge': touch_edge,
                'area': area,
                'volume': volume,
                'ellip_volume': ellip_volume, 
                'iscrowd': 0,
                'category_id': 1,
                'segmentation': [segmentation]
            }
            coco_format['annotations'].append(annotation)
            annotation_id += 1

    return coco_format

def list_files_full_path(directory):
    """
    Returns a list of files with their full paths in the specified directory.
    """
    full_paths = []
    for filename in os.listdir(directory):
        full_path = os.path.join(directory, filename)
        full_paths.append(full_path)
    return full_paths