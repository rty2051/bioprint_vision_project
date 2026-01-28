#!/usr/bin/env python3.11
"""
File: save_filtered_contour_lengths.py
Author: Ryan Yam, @rty2051
Description: Detects shapes, merges nearly-continuous contours, labels them,
             filters contours based on pixel length, saves labeled image and pixel lengths.
"""

import cv2 as cv
import cv2
import numpy as np

# IMAGE_PATH = "C:\\Users\\Prometheus\\Downloads\\bioprint_vision_project\\images\\better_res_orange.jpg"
IMAGE_PATH = "C:\\Users\\Prometheus\\Downloads\\bioprint_vision_project\\images\\better_res_orange.jpg"
OUTPUT_IMAGE = "LABELED_CONTOURS.png"
OUTPUT_TXT = "CONTOUR_LENGTHS.txt"

def main():
    img = cv.imread(IMAGE_PATH)

    # Crop to build area
    x1, y1 = 3900, 3
    x2, y2 = 880, 2479
    x_min = min(x1, x2)
    x_max = max(x1, x2)
    y_min = min(y1, y2)
    y_max = max(y1, y2)
    crop = img[y_min:y_max, x_min:x_max]

    # Convert to grayscale and blur
    gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (9, 9), 0)

    # Canny edge detection
    edges = cv.Canny(blurred, 0, 55)

    # Create binary mask from edges
    mask = edges.copy()
    mask[mask > 0] = 255

    # Dilate mask to connect nearly-touching contours
    kernel = np.ones((7, 7), np.uint8)  # 3x3 kernel connects 1-pixel gaps
    dilated = cv.dilate(mask, kernel, iterations=1)

    # Find contours on dilated mask
    contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    rng = np.random.default_rng(42)
    output = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)

    # Dictionary to store pixel lengths of valid contours
    contour_lengths = {}

    for idx, cnt in enumerate(contours):
        # Pixel length of contour
        pixel_length = len(cnt)

        # Filter: remove contours with length ~50 ± 1 or > 1000
        if 49 <= pixel_length <= 80 or pixel_length > 1000 or pixel_length < 40:
            continue

        # Store pixel length
        contour_lengths[idx] = pixel_length

        # Draw contour in random color
        color = rng.integers(0, 256, size=3).tolist()
        cv.drawContours(output, [cnt], -1, color, 2)

        # Label contour at centroid
        M = cv.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = cnt[0][0]
        cv.putText(output, str(idx), (cX, cY), cv.FONT_HERSHEY_SIMPLEX,
                   1, (255, 255, 255), 2, cv.LINE_AA)

    # Save the labeled image
    cv.imwrite(OUTPUT_IMAGE, output)
    print(f"Labeled image saved as '{OUTPUT_IMAGE}'")

    # Save filtered contour lengths to TXT
    with open(OUTPUT_TXT, "w") as f:
        for idx, length in contour_lengths.items():
            f.write(f"Contour {idx}: {length} pixels\n")

    print(f"Filtered contour lengths saved as '{OUTPUT_TXT}'")

    return contour_lengths

if __name__ == "__main__":
    lengths_dict = main()
