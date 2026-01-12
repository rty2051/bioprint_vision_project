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


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]
    union = areaA + areaB - inter

    return inter / union if union > 0 else 0

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

    # Find contours
    contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Filter overlapping contours
    filtered = []
    boxes = [cv.boundingRect(c) for c in contours]
    for i, cnt in enumerate(contours):
        keep = True
        for j in range(i):
            if iou(boxes[i], boxes[j]) > 0.9:
                keep = False
                break
        if keep:
            filtered.append(cnt)

    true_squares = []
    rng = np.random.default_rng(42)
    output = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
    for cnt in filtered:
        # Approximate contour to polygon
        peri = cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, 0.05 * peri, True)

        # Step 1 + 2: check 4 corners and convexity
        if len(approx) == 4 and cv.isContourConvex(approx):
            # Step 3: check side lengths
            pts = approx.reshape(4, 2)
            sides = [np.linalg.norm(pts[i] - pts[(i + 1) % 4]) for i in range(4)]
            if max(sides) / min(sides) <= 1.2:  # roughly equal sides
                true_squares.append(approx)

                # Draw immediately in random color (optional)
                color = rng.integers(0, 256, size=3).tolist()
                cv.drawContours(output, [approx], -1, color, 2)

#    # Draw Squares
#     output = cv.cvtColor(edges, cv.COLOR_GRAY2BGR)
#     rng = np.random.default_rng(42)  # fixed seed (optional)
#     for cnt in filtered:
#         color = rng.integers(0, 256, size=3).tolist()
#         cv.drawContours(output, [cnt], -1, color, 2)

    # Resize ONLY for display
    output_resized = cv.resize(
        output, None,
        fx=0.5, fy=0.5,
        interpolation=cv.INTER_LINEAR
    )
    cv.imshow("Edge Lengths", output_resized)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite("labeled_contours.png", output)


    # display = cv.resize(edges, None, fx=0.5, fy=0.5, interpolation=cv.INTER_LINEAR)    
    # cv.imshow("Test", display)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

main()