"""Camera calibration using OpenCV with a chessboard pattern."""

import cv2
import numpy as np
import glob
from pathlib import Path

IMAGE_PATH = "/home/ryam/Desktop/bioprint_vision_project/images"
CHESS_PATH = IMAGE_PATH + "/chessboard"
CALIBRATION_OUTPUT = "calibration/calibration.npz"

def calibrate_camera(
    chess_path: str,
    checkerboard: tuple[int, int] = (9, 6),
    square_size_mm: float = 1.0,
    output_npz: str = "calibration.npz",
    visualize: bool = False,
) -> dict:
    """
    Calibrate a camera using chessboard images.

    Args:
        chess_path:          Directory containing chessboard .jpg images.
        checkerboard:        Number of inner corners (cols, rows). Default (9, 6).
        square_size_mm:      Physical size of each square in mm. Default 1.0 (unit-less).
        output_npz:          Path to save calibration results (.npz).
        test_image_path:     Optional path to a single image to undistort.
                             Defaults to the first calibration image.
        undistorted_output:  Path to save the undistorted test image.
        visualize:           If True, display detected corners during processing.

    Returns:
        dict with keys:
            camera_matrix, dist_coeffs, rvecs, tvecs,
            reprojection_error, detected, failed
    """
    # ── 1. Prepare object points ───────────────────────────────────────────────
    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    objp *= square_size_mm

    objpoints, imgpoints = [], []
    image_size = None
    detected = failed = 0

    # ── 2. Load images and detect corners ─────────────────────────────────────
    images = glob.glob(str(Path(chess_path) / "*.jpg"))
    if not images:
        raise FileNotFoundError(f"No .jpg images found in: {chess_path}")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            print(f"[WARN] Could not read {fname}, skipping.")
            failed += 1
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]  # (width, height)

        ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)

        if ret:
            detected += 1
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            if visualize:
                cv2.drawChessboardCorners(img, checkerboard, corners2, ret)
                cv2.imshow("Corners", img)
                cv2.waitKey(500)
        else:
            failed += 1
            print(f"[INFO] Chessboard not detected in {fname}")

    if visualize:
        cv2.destroyAllWindows()

    if detected < 3:
        raise RuntimeError(
            f"Too few valid images for calibration ({detected} detected, need ≥ 3)."
        )

    print(f"\nDetected corners in {detected}/{detected + failed} images.")

    # ── 3. Calibrate ──────────────────────────────────────────────────────────
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )

    print("Camera Matrix:\n", camera_matrix)
    print("\nDistortion Coefficients:\n", dist_coeffs)
    print(f"\nReprojection Error: {ret:.4f}  (good < 0.5, acceptable < 1.0)")

    # ── 4. Save results ────────────────────────────────────────────────────────
    np.savez(output_npz, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print(f"\nCalibration saved to: {output_npz}")


"""Compute homography from pixel coordinates to real-world mm using chessboard corners."""

import cv2
import numpy as np


def compute_homography(
    calibration_npz: str,
    reference_image: str,
    checkerboard: tuple[int, int] = (9, 6),
    square_size_mm: float = 20.447,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute a homography matrix mapping pixel coordinates to real-world mm.

    Args:
        calibration_npz:  Path to .npz file with camera_matrix & dist_coeffs.
        reference_image:  Path to a chessboard image for homography estimation.
        checkerboard:     Inner corner count (cols, rows). Default (9, 6).
        square_size_mm:   Physical square size in mm. Default 20.447.

    Returns:
        H               (3×3) homography matrix  (pixel → mm)
        camera_matrix   (3×3) camera intrinsics
        dist_coeffs     distortion coefficients
    """
    # ── 1. Load calibration ────────────────────────────────────────────────────
    data = np.load(calibration_npz)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]

    # ── 2. Build real-world object points (Z = 0 plane) ───────────────────────
    objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard[0], 0:checkerboard[1]].T.reshape(-1, 2)
    objp *= square_size_mm
    objp_2d = objp[:, :2]  # drop Z; board is flat

    # ── 3. Detect corners in the reference image ───────────────────────────────
    img = cv2.imread(reference_image)
    if img is None:
        raise FileNotFoundError(f"Could not read reference image: {reference_image}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, checkerboard, None)
    if not ret:
        raise RuntimeError("Chessboard corners not found in the reference image.")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # ── 4. Undistort corner pixels ─────────────────────────────────────────────
    corners_undistorted = cv2.undistortPoints(
        corners2, camera_matrix, dist_coeffs, P=camera_matrix
    )

    # ── 5. Compute homography (pixel → mm) ────────────────────────────────────
    H, mask = cv2.findHomography(corners_undistorted, objp_2d)
    if H is None:
        raise RuntimeError("Homography computation failed.")

    inliers = int(mask.sum())
    print(f"Homography computed — inliers: {inliers}/{len(mask)}")
    print("Homography Matrix H:\n", H)

    return H, camera_matrix, dist_coeffs


def pixel_to_mm(
    pixel_point: tuple[float, float],
    H: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> tuple[float, float]:
    """
    Convert a single (x, y) pixel coordinate to real-world mm.

    Args:
        pixel_point:    (x, y) pixel coordinate.
        H:              3×3 homography matrix from compute_homography().
        camera_matrix:  Camera intrinsics.
        dist_coeffs:    Distortion coefficients.

    Returns:
        (X_mm, Y_mm) real-world position in millimetres.
    """
    pt = np.array([[[pixel_point[0], pixel_point[1]]]], dtype=np.float32)
    pt_undistorted = cv2.undistortPoints(pt, camera_matrix, dist_coeffs, P=camera_matrix)

    pt_h = np.array([pt_undistorted[0][0][0], pt_undistorted[0][0][1], 1.0])
    mm = H @ pt_h
    mm /= mm[2]  # normalise homogeneous coordinate
    return float(mm[0]), float(mm[1])


def pixels_to_mm_batch(
    pixel_points: np.ndarray,
    H: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    """
    Convert an array of pixel coordinates to real-world mm (vectorised).

    Args:
        pixel_points:   (N, 2) array of (x, y) pixel coordinates.
        H:              3×3 homography matrix from compute_homography().
        camera_matrix:  Camera intrinsics.
        dist_coeffs:    Distortion coefficients.

    Returns:
        (N, 2) array of (X_mm, Y_mm) real-world positions.
    """
    pts = pixel_points.astype(np.float32).reshape(-1, 1, 2)
    pts_undistorted = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, P=camera_matrix)
    pts_undistorted = pts_undistorted.reshape(-1, 2)

    # Convert to homogeneous, apply H, normalise
    ones = np.ones((len(pts_undistorted), 1), dtype=np.float64)
    pts_h = np.hstack([pts_undistorted, ones])          # (N, 3)
    mm_h = (H @ pts_h.T).T                              # (N, 3)
    mm = mm_h[:, :2] / mm_h[:, 2:3]                    # (N, 2)
    return mm


# ── Example usage ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    results = calibrate_camera(
        chess_path=CHESS_PATH,
        checkerboard=(9, 6),
        square_size_mm=25.0,
        output_npz=CALIBRATION_OUTPUT,
        visualize=True,
    )

    H, camera_matrix, dist_coeffs = compute_homography(
        calibration_npz=CALIBRATION_OUTPUT,
        reference_image=CHESS_PATH + "/chessboard2_20260305.jpg",
        checkerboard=(9, 6),
        square_size_mm=20.447,
    )

    # Single point
    px, py = 320, 240
    x_mm, y_mm = pixel_to_mm((px, py), H, camera_matrix, dist_coeffs)
    print(f"Pixel ({px}, {py})  →  ({x_mm:.3f} mm, {y_mm:.3f} mm)")

    # Batch conversion
    points = np.array([[320, 240], [640, 480], [100, 150]], dtype=np.float32)
    results = pixels_to_mm_batch(points, H, camera_matrix, dist_coeffs)
    for (px, py), (xmm, ymm) in zip(points, results):
        print(f"Pixel ({px:.0f}, {py:.0f})  →  ({xmm:.3f} mm, {ymm:.3f} mm)")