"""
calibrate.py
------------
Camera calibration using a chessboard pattern.

Can be used as a module:
    from calibrate import run_calibration
    run_calibration("images/chessboard", "calibration/calibration_data.npz")

Or run directly from the command line:
    python calibrate.py --input images/chessboard --output calibration/calibration_data.npz
"""

import argparse
import glob
import os
import cv2
import numpy as np

def load_images(image_folder: str) -> list[np.ndarray]:
    """
    Load all .jpg images from *image_folder*. Returns (path, image) pairs.
    
    @params:
        image_folder:  Path to a folder containing .jpg images.
    
    @returns:
        List of (image_path, image_array) tuples.
    """
    paths = glob.glob(os.path.join(image_folder, "*.jpg"))
    print(f"Found {len(paths)} image(s) in '{image_folder}'.")

    images = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  [WARN] Could not read {path} — skipping.")
        else:
            images.append((path, img))
    return images

def detect_corners(
    image: np.ndarray,
    chessboard_size: tuple[int, int],
) -> tuple[bool, np.ndarray | None]:
    """
    Detect sub-pixel chessboard corners in *image*.

    @params:
        image:            BGR image.
        chessboard_size:  (cols, rows) inner corners.

    @returns:
        (success, refined_corners)  — corners is None when detection fails.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCornersSB(gray, chessboard_size)

    if not ret:
        return False, None

    refined = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
    )
    return True, refined

def compute_pixel_distances(
    corners: np.ndarray,
    chessboard_size: tuple[int, int],
) -> list[float]:
    """
    Measure horizontal and vertical distances (in pixels) between every
    adjacent pair of inner corners on the board.

    @params:
        corners:          Refined corner array from cv2.cornerSubPix.
        chessboard_size:  (cols, rows) inner corners.

    @returns:
        List of pixel distances.
    """
    cols, rows = chessboard_size
    distances = []

    # Horizontal neighbours
    for row in range(rows):
        for col in range(cols - 1):
            idx = row * cols + col
            p1 = corners[idx][0]
            p2 = corners[idx + 1][0]
            distances.append(float(np.linalg.norm(p1 - p2)))

    # Vertical neighbours
    for row in range(rows - 1):
        for col in range(cols):
            idx = row * cols + col
            p1 = corners[idx][0]
            p2 = corners[idx + cols][0]
            distances.append(float(np.linalg.norm(p1 - p2)))

    return distances

def build_object_points(
    chessboard_size: tuple[int, int],
    square_size: float,
) -> np.ndarray:
    """
    Build the 3-D object point array for one chessboard view.

    @params:
        chessboard_size:  (cols, rows) inner corners.
        square_size:      Physical size of one square in real-world units (e.g. mm).

    @returns:
        Array of shape (cols*rows, 3) in float32.
    """
    cols, rows = chessboard_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


def calibrate_camera(
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, list, list]:
    """
    Run cv2.calibrateCamera and return the intrinsic parameters.

    @params:
        objpoints:   List of 3-D object point arrays (one per image).
        imgpoints:   List of 2-D image point arrays (one per image).
        image_size:  (width, height) of the calibration images.

    @returns:
        camera_matrix, dist_coeffs, rvecs, tvecs
    """
    _, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None
    )
    return camera_matrix, dist_coeffs, rvecs, tvecs

def save_calibration(
    output_path: str,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvecs: list,
    tvecs: list,
    pixel_to_mm: float,
) -> None:
    """
    Save all calibration data to a .npz file.
    @params:
        output_path:   Path for the output .npz file.
        camera_matrix: Intrinsic camera matrix.
        dist_coeffs:   Distortion coefficients.
        rvecs:         List of rotation vectors (one per image).
        tvecs:         List of translation vectors (one per image).
        pixel_to_mm:   Scale factor from pixels to mm.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez(
        output_path,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rvecs=rvecs,
        tvecs=tvecs,
        pixel_to_mm=pixel_to_mm,
    )
    print(f"[Saved] Calibration data → {output_path}")

def run_calibration(
    image_folder: str,
    output_file: str,
    chessboard_size: tuple[int, int] = (6, 4),
    square_size: float = 13.0,
) -> dict:
    """
    Full calibration pipeline.

    @params:
        image_folder:    Folder containing chessboard .jpg images.
        output_file:     Path for the output .npz file.
        chessboard_size: (cols, rows) inner corners of the chessboard.
        square_size:     Physical size of one square in real-world units (e.g. mm).

    @returns:
        Dictionary with keys: camera_matrix, dist_coeffs, rvecs, tvecs, pixel_to_mm.

    @exceptions:
        RuntimeError: If no corners or distances were detected.
    """
    objp_template = build_object_points(chessboard_size, square_size)

    objpoints:       list[np.ndarray] = []
    imgpoints:       list[np.ndarray] = []
    pixel_distances: list[float]      = []
    image_size:      tuple | None     = None

    for path, image in load_images(image_folder):
        success, corners = detect_corners(image, chessboard_size)
        if not success:
            print(f"  [FAIL] No corners found in {path}")
            continue

        print(f"  [OK]   Corners detected in {path}")
        objpoints.append(objp_template)
        imgpoints.append(corners)
        pixel_distances.extend(compute_pixel_distances(corners, chessboard_size))
        image_size = (image.shape[1], image.shape[0])  # (width, height)

    if not objpoints:
        raise RuntimeError("No chessboard corners were detected in any image.")
    if not pixel_distances:
        raise RuntimeError("No pixel distances could be computed.")

    # --- Camera calibration --------------------------------------------------
    camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(
        objpoints, imgpoints, image_size
    )
    print("\nCalibration successful!")

    # --- Pixel → mm scale factor ---------------------------------------------
    avg_pixel_dist = float(np.mean(pixel_distances))
    pixel_to_mm    = square_size / avg_pixel_dist

    print(f"Average pixel distance : {avg_pixel_dist:.4f} px")
    print(f"Pixel-to-mm scale      : {pixel_to_mm:.6f} mm/px")

    # --- Persist & return ----------------------------------------------------
    save_calibration(output_file, camera_matrix, dist_coeffs, rvecs, tvecs, pixel_to_mm)

    return {
        "camera_matrix": camera_matrix,
        "dist_coeffs":   dist_coeffs,
        "rvecs":         rvecs,
        "tvecs":         tvecs,
        "pixel_to_mm":   pixel_to_mm,
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate a camera using chessboard images."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        metavar="IMAGE_FOLDER",
        help="Folder containing chessboard .jpg images.",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        metavar="OUTPUT_NPZ",
        help="Path for the output calibration .npz file.",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=6,
        help="Number of inner corner columns on the chessboard (default: 6).",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=4,
        help="Number of inner corner rows on the chessboard (default: 4).",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=13.0,
        help="Physical size of one square in mm (default: 13.0).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_calibration(
        image_folder    = args.input,
        output_file     = args.output,
        chessboard_size = (args.cols, args.rows),
        square_size     = args.square_size,
    )