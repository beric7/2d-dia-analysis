# Copyright 2025 The MITRE Corporation

import cv2
import numpy as np
from utils import smallestenclosingcircle as sec
import feret
from math import pi, sin, cos
from typing import Tuple, Optional

def compute_sphericity(grainMask: np.ndarray, area: float) -> Tuple[float, Tuple[float, float, float]]:
    """
    Compute sphericity of a grain based on its contour mask and area.

    Args:
        grainMask (np.ndarray): Contour mask of the grain (OpenCV contour format).
        area (float): Area of the grain.

    Returns:
        -> Tuple[float, Tuple[float, float, float]]:
            - sphericity (float): Ratio of grain area to area of smallest enclosing circle.
            - circle (tuple): (x, y, radius) of smallest enclosing circle.
    """
    pixelCoords = [(val[0][0], val[0][1]) for val in grainMask]
    circle = sec.make_circle(pixelCoords)
    radius = circle[2]
    circleArea = np.pi * radius * radius
    sphericity = area / circleArea
    return sphericity, circle

def compute_sphericity_all(grainMask: np.ndarray, area: float) -> Tuple[float, Tuple[float, float, float]]:
    """
    Compute sphericity for a binary mask (all pixels) and area.

    Args:
        grainMask (np.ndarray): Binary mask of the grain (2D array).
        area (float): Area of the grain.

    Returns:
        -> Tuple[float, Tuple[float, float, float]]:
            - sphericity (float): Ratio of grain area to area of smallest enclosing circle.
            - circle (tuple): (x, y, radius) of smallest enclosing circle.
    """
    pixelCoords = [(idy, idx) for idx, row in enumerate(grainMask) for idy, val in enumerate(row) if val]
    circle = sec.make_circle(pixelCoords)
    radius = circle[2]
    circleArea = np.pi * radius * radius
    sphericity = area / circleArea
    return sphericity, circle

def compute_feret_diameters(contour: np.ndarray) -> Tuple[float, float]:
    """
    Compute Feret minimum and maximum diameters using OpenCV's minAreaRect.

    Args:
        contour (np.ndarray): Contour of the grain.

    Returns:
        -> Tuple[float, float]:
            - feret_min (float): Minimum Feret diameter.
            - feret_max (float): Maximum Feret diameter.
    """
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)  # Convert to integer
    feret_min = min(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2]))
    feret_max = max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2]))
    return feret_min, feret_max

def compute_volume(feret_result, PIXEL_TO_MM: float) -> float:
    """
    Compute grain volume using Feret diameters and their angles.

    Args:
        feret_result (feret.FeretResult): Result object from feret.calc().
        PIXEL_TO_MM (float): Conversion factor from pixels to millimeters.

    Returns:
        -> float: Estimated grain volume.
    """
    res = feret_result
    fmin = res.minf * PIXEL_TO_MM
    fmax = res.maxf * PIXEL_TO_MM

    theta = abs(res.maxf_angle - res.minf_angle)
    if theta > 2 * pi:
        raise Exception("Theta value too large")

    if theta > pi:
        theta = 2 * pi - theta
    if theta > pi / 2:
        theta = pi - theta

    volume = (4 * pi / 3) * fmax**3 * fmin**2 * sin(theta)**2 / (fmax**2 - fmin**2 * cos(theta)**2)
    return volume

def compute_volume_simple(feret_min: float, feret_max: float, PIXEL_TO_MM: float) -> float:
    """
    Compute grain volume using Feret min/max diameters, assuming perpendicular axes.

    Args:
        feret_min (float): Minimum Feret diameter (pixels).
        feret_max (float): Maximum Feret diameter (pixels).
        PIXEL_TO_MM (float): Conversion factor from pixels to millimeters.

    Returns:
        -> float: Estimated grain volume.
    """
    fmin = feret_min * PIXEL_TO_MM
    fmax = feret_max * PIXEL_TO_MM

    # Assume theta is pi/2 for simplicity (perpendicular diameters)
    theta = pi / 2

    volume = (4 * pi / 3) * fmax**3 * fmin**2 * sin(theta)**2 / (fmax**2 - fmin**2 * cos(theta)**2)
    return volume

def ellipsoidal_volume(
    fmin: float,
    fmax: float,
    PIXEL_TO_MM: float,
    third_dimension = None
) -> float:
    """
    Compute ellipsoidal volume from Feret min/max diameters and an optional third dimension.

    Args:
        fmin (float): Minimum Feret diameter (pixels).
        fmax (float): Maximum Feret diameter (pixels).
        PIXEL_TO_MM (float): Conversion factor from pixels to millimeters.
        third_dimension (Optional[float]): Third axis length (mm). If None, average of fmin and fmax.

    Returns:
        -> float: Estimated ellipsoidal volume.
    """
    fmin = fmin * PIXEL_TO_MM
    fmax = fmax * PIXEL_TO_MM
    if third_dimension is None:
        third_dimension = (fmin + fmax) / 2

    a = fmin / 2
    b = fmax / 2
    c = third_dimension / 2

    volume = (4 / 3) * np.pi * a * b * c
    return volume
