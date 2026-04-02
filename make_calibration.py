import cv2
import numpy as np
import glob
import os

# === SETTINGS ===
chessboard_size = (7, 5)  # number of inner corners per row and column
square_size = 1.0         # real-world square size (arbitrary units, e.g. meters)

image_folder = "images/chessboard"
output_file = "calibration/calibration_data.npz"

# === PREPARE OBJECT POINTS ===
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Get all images
images = glob.glob(os.path.join(image_folder, "*.jpg"))

print(f"Found {len(images)} images.")

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Could not read {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)

        # Refine corner locations
        corners2 = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        imgpoints.append(corners2)

        print(f"Detected corners in {fname}")
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
print("Camera matrix:\n", camera_matrix)
print("Distortion coefficients:\n", dist_coeffs)

# === SAVE TO NPZ ===
np.savez(
    output_file,
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    rvecs=rvecs,
    tvecs=tvecs
)

print(f"\nSaved calibration data to {output_file}")