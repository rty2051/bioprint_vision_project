#!/usr/bin/env python3.11.0
"""
File: regular_filament.py
Author: Ryan Yam, @rty2051
Description: Scan and detect the printed object shape using regular filament. Counts pixels to determine dimensions
"""

import cv2 as cv
import cv2
import numpy as np


IMAGE_PATH = "C:\\Users\\Prometheus\\Downloads\\bioprint_vision_project\\images\\no_back_white.jpg"

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

    # Convert to grayscale
    gray = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)

    # (Optional but recommended) Reduce noise
    blurred = cv.GaussianBlur(gray, (11, 11), 0)

    # Canny edge detection
    edges = cv.Canny(blurred, 50, 150)

    contours, _ = cv.findContours(edges, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)

    # Pixel Counting for Edges
    edge_lengths = []
    for i, cnt in enumerate(contours):
        length = cv.arcLength(cnt, closed=False)
        edge_lengths.append(length)
        print(f"Edge {i}: length = {length:.2f} pixels")

    # Display
    output = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    for i, cnt in enumerate(contours):
        cv.drawContours(output, [cnt], -1, (0, 255, 0), 1)
        x, y = cnt[0][0]
        cv.putText(output, f"{i}: {int(cv.arcLength(cnt, False))}",
                (x, y), cv.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 0, 255), 1)
    # Resize ONLY for display
    output_resized = cv.resize(
        output, None,
        fx=0.7, fy=0.7,
        interpolation=cv.INTER_LINEAR
    )
    cv.imshow("Edge Lengths", output_resized)
    cv.waitKey(0)
    cv.destroyAllWindows()


    # display = cv.resize(edges, None, fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)    
    # cv.imshow("Test", display)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

main()