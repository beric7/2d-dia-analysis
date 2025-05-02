# grain_analysis.py
import cv2
import numpy as np
from utils import smallestenclosingcircle as sec
import feret
from math import pi, sin, cos

def compute_sphericity(grainMask, area):
    pixelCoords = [(val[0][0], val[0][1]) for val in grainMask]
    circle = sec.make_circle(pixelCoords)
    radius = circle[2]
    circleArea = np.pi * radius * radius
    sphericity = area / circleArea
    return sphericity, circle

def compute_sphericity_all(grainMask, area):
    pixelCoords = [(idy, idx) for idx, row in enumerate(grainMask) for idy, val in enumerate(row) if val]
    circle = sec.make_circle(pixelCoords)
    radius = circle[2]
    circleArea = np.pi * radius * radius
    sphericity = area / circleArea
    return sphericity, circle

def compute_feret_diameters(contour):
    """
    Compute Feret minimum and maximum diameters using OpenCV's minAreaRect.
    """
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)  # Convert to integer
    feret_min = min(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2]))
    feret_max = max(np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2]))
    return feret_min, feret_max

def compute_volume(feret_result, PIXEL_TO_MM):
    res = feret_result
    fmin = res.minf * PIXEL_TO_MM
    fmax = res.maxf * PIXEL_TO_MM

    theta = res.maxf_angle - res.minf_angle
    theta = abs(theta)
    if theta > 2 * pi:
        raise Exception("Theta value too large")

    if theta > pi:
        theta = 2 * pi - theta
    if theta > pi / 2:
        theta = pi - theta

    volume = (4 * pi / 3) * fmax**3 * fmin**2 * sin(theta)**2 / (fmax**2 - fmin**2 * cos(theta)**2)
    return volume

def compute_volume_simple(feret_min, feret_max, PIXEL_TO_MM):
    fmin = feret_min * PIXEL_TO_MM
    fmax = feret_max * PIXEL_TO_MM

    # Assume theta is pi/2 for simplicity (perpendicular diameters)
    theta = pi / 2

    volume = (4 * pi / 3) * fmax**3 * fmin**2 * sin(theta)**2 / (fmax**2 - fmin**2 * cos(theta)**2)
    return volume

def ellipsoidal_volume(fmin, fmax, PIXEL_TO_MM, third_dimension=None):
    fmin = fmin * PIXEL_TO_MM
    fmax = fmax * PIXEL_TO_MM
    if third_dimension is None:
        third_dimension = (fmin + fmax) / 2

    a = fmin / 2
    b = fmax / 2
    c = third_dimension / 2

    volume = (4 / 3) * np.pi * a * b * c
    return volume
