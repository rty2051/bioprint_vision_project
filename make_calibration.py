import cv2
import numpy as np
import glob
import os

# === SETTINGS ===
chessboard_size = (6, 4)   # inner corners
square_size = 13.0          # REAL size of square (e.g., mm!)

image_folder = "images/chessboard"
output_file = "calibration/calibration_data.npz"

# === PREPARE OBJECT POINTS ===
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

objpoints = []
imgpoints = []

# NEW: store pixel distances
pixel_distances = []

images = glob.glob(os.path.join(image_folder, "*.jpg"))
print(f"Found {len(images)} images.")

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Could not read {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ret, corners = cv2.findChessboardCorners(
    #     gray,
    #         chessboard_size,
    #         cv2.CALIB_CB_ADAPTIVE_THRESH +
    #         cv2.CALIB_CB_NORMALIZE_IMAGE +
    #         cv2.CALIB_CB_FAST_CHECK
    #     )
    ret, corners = cv2.findChessboardCornersSB(gray, chessboard_size)

    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        imgpoints.append(corners2)

        print(f"Detected corners in {fname}")

        # === COMPUTE PIXEL DISTANCES ===
        # Horizontal distances
        for i in range(chessboard_size[1]):
            for j in range(chessboard_size[0] - 1):
                idx = i * chessboard_size[0] + j
                p1 = corners2[idx][0]
                p2 = corners2[idx + 1][0]
                dist = np.linalg.norm(p1 - p2)
                pixel_distances.append(dist)

        # Vertical distances
        for i in range(chessboard_size[1] - 1):
            for j in range(chessboard_size[0]):
                idx = i * chessboard_size[0] + j
                p1 = corners2[idx][0]
                p2 = corners2[idx + chessboard_size[0]][0]
                dist = np.linalg.norm(p1 - p2)
                pixel_distances.append(dist)

    else:
        print(f"Failed to detect corners in {fname}")

# === CALIBRATION ===
if len(objpoints) == 0:
    raise RuntimeError("No chessboard corners were detected in any image.")

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    gray.shape[::-1],
    None,
    None
)

print("\nCalibration successful!")

# === COMPUTE PIXEL TO MM ===
if len(pixel_distances) == 0:
    raise RuntimeError("No pixel distances computed.")

avg_pixel_distance = np.mean(pixel_distances)

pixel_to_mm = square_size / avg_pixel_distance

print(f"Average pixel distance: {avg_pixel_distance:.4f} px")
print(f"Pixel to mm scale: {pixel_to_mm:.6f} mm/px")

# === SAVE ===
np.savez(
    output_file,
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    rvecs=rvecs,
    tvecs=tvecs,
    pixel_to_mm=pixel_to_mm
)

print(f"\nSaved calibration data to {output_file}")