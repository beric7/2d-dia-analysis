import cv2
import numpy as np

# Load the image
image = cv2.imread('/projects/OLIVINE/ebianchi/juniorstav/temp-11262024112518-39.jpg', 0)
otsu_threshold, image_result = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print(f'threshold is: {otsu_threshold}')
inverted_image = cv2.bitwise_not(image_result)
print(inverted_image)
cv2.imwrite('./inverted.jpg', inverted_image)
height, width = image.shape[:2]

contours, _ = cv2.findContours(inverted_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate through each contour and save them as separate images
for i, contour in enumerate(contours):
    # Create a mask for the current contour
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

    # Save the cropped image
    filename = f'contour_{i}'
    if touches_edge:
        filename += '_edge'
    filename += '.jpg'
    cv2.imwrite(filename, cropped)

    # Print if the contour touches the edge
    if touches_edge:
        print(f"Contour {i} touches the edge of the original image.")

print("Contours have been saved as separate images.")