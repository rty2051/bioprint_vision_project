"""
object_detection.py
-------------------
Detects the dominant-colored object in an image and (optionally) reports
its area in mm² using a camera calibration file.

Usage
-----
# Basic detection only
python object_detection.py --input path/to/image.jpg --output-dir path/to/out/

# With calibration (area in mm²)
python object_detection.py \
    --input path/to/image.jpg \
    --output-dir path/to/out/ \
    --calibration path/to/calibration_data.npz

Optional tuning flags
---------------------
  --h-std-factor   float  Hue tolerance in std-devs        (default: 2.0)
  --s-std-factor   float  Saturation tolerance in std-devs (default: 1.5)
  --v-std-factor   float  Value tolerance in std-devs      (default: 2.0)
  --min-saturation int    Minimum saturation for pre-filter (default: 60)
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Dominant-color sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_dominant_hue(hsv_image: np.ndarray, min_saturation: int = 60) -> int:
    """
    Return the dominant hue (0–179) of sufficiently saturated pixels.

    The hue histogram is wrapped to avoid split peaks near 0/179 (reds),
    then the peak bin is returned.

    Args:
        hsv_image:      Full HSV image (H in [0,179]).
        min_saturation: Ignore pixels below this saturation value.

    Returns:
        Dominant hue value in [0, 179].
    """
    h_channel = hsv_image[:, :, 0]
    s_channel = hsv_image[:, :, 1]

    saturated = s_channel >= min_saturation
    if saturated.sum() < 50:
        raise RuntimeError(
            "Too few saturated pixels to determine a dominant color. "
            "Try lowering --min-saturation."
        )

    hues = h_channel[saturated].astype(np.int32)

    # Build a doubled histogram to handle wrap-around at 0/179
    hist = np.bincount(hues, minlength=180)
    doubled = np.concatenate([hist, hist])
    # Smooth with a 15-bin window
    kernel = np.ones(15) / 15
    smoothed = np.convolve(doubled, kernel, mode="same")
    peak = int(np.argmax(smoothed[:180]))  # keep result in [0,179]

    print(f"[DominantHue] peak hue = {peak}  (from {saturated.sum()} saturated pixels)")
    return peak


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Adaptive HSV bounds (color-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def compute_adaptive_bounds(
    hsv_image: np.ndarray,
    dominant_hue: int,
    hue_window: int = 20,
    min_saturation: int = 60,
    h_std_factor: float = 2.0,
    s_std_factor: float = 1.5,
    v_std_factor: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute tight HSV bounds centered on the dominant hue.

    1. Pre-filter pixels within ±hue_window of the dominant hue.
    2. Measure the HSV distribution of those pixels.
    3. Return mean ± k*std bounds, clamped to valid ranges.

    Args:
        hsv_image:      Full HSV image (H in [0,179]).
        dominant_hue:   The hue to center the search around.
        hue_window:     Half-width of the pre-filter hue band.
        min_saturation: Minimum saturation for the pre-filter.
        h_std_factor:   Std-dev multiplier for Hue.
        s_std_factor:   Std-dev multiplier for Saturation.
        v_std_factor:   Std-dev multiplier for Value.

    Returns:
        (lower, upper) as np.ndarray in OpenCV HSV scale.
    """
    h, s, v = (
        hsv_image[:, :, 0].astype(np.int32),
        hsv_image[:, :, 1],
        hsv_image[:, :, 2],
    )

    # Hue distance with wrap-around
    hue_dist = np.abs(h - dominant_hue)
    hue_dist = np.minimum(hue_dist, 180 - hue_dist)

    pre_mask = (hue_dist <= hue_window) & (s >= min_saturation)

    if pre_mask.sum() < 50:
        print(
            "[WARN] Very few pixels matched dominant hue pre-filter. "
            "Falling back to a ±hue_window hard window."
        )
        lo_h = max(0,   dominant_hue - hue_window)
        hi_h = min(179, dominant_hue + hue_window)
        return np.array([lo_h, min_saturation, 52]), np.array([hi_h, 255, 235])

    h_vals = h[pre_mask].astype(float)
    s_vals = s[pre_mask].astype(float)
    v_vals = v[pre_mask].astype(float)

    h_mean, h_std = h_vals.mean(), h_vals.std()
    s_mean, s_std = s_vals.mean(), s_vals.std()
    v_mean, v_std = v_vals.mean(), v_vals.std()

    print(
        f"[AdaptiveBounds] H: {h_mean:.1f} ± {h_std:.1f}  "
        f"S: {s_mean:.1f} ± {s_std:.1f}  "
        f"V: {v_mean:.1f} ± {v_std:.1f}"
    )

    lower = np.array([
        max(0,   int(h_mean - h_std_factor * h_std)),
        max(0,   int(s_mean - s_std_factor * s_std)),
        max(0,   int(v_mean - v_std_factor * v_std)),
    ])
    upper = np.array([
        min(179, int(h_mean + h_std_factor * h_std)),
        255,
        min(255, int(v_mean + v_std_factor * v_std)),
    ])

    print(f"[AdaptiveBounds] lower={lower}  upper={upper}")
    return lower, upper


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Object detection
# ─────────────────────────────────────────────────────────────────────────────

def object_detection(
    image_path: str | Path,
    output_dir: str | Path,
    min_saturation: int = 60,
    h_std_factor: float = 2.0,
    s_std_factor: float = 1.5,
    v_std_factor: float = 2.0,
) -> list:
    """
    Detect the dominant-colored object in *image_path* and write diagnostic
    images to *output_dir*.

    Returns the list of contours found (same order as cv2.findContours).
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # ── Determine the dominant hue automatically ───────────────────────────
    dominant_hue = sample_dominant_hue(hsv, min_saturation=min_saturation)

    # ── Compute tight HSV bounds around that hue ───────────────────────────
    lower, upper = compute_adaptive_bounds(
        hsv,
        dominant_hue=dominant_hue,
        min_saturation=min_saturation,
        h_std_factor=h_std_factor,
        s_std_factor=s_std_factor,
        v_std_factor=v_std_factor,
    )

    mask = cv2.inRange(hsv, lower, upper)

    # ── Morphological clean-up ─────────────────────────────────────────────
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((1,  1),  np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

    cv2.imwrite(str(output_dir / "mask.png"), mask)

    # ── Contour detection ──────────────────────────────────────────────────
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[1: ] # skip the outermost contour which is just the image border

    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 2)
    cv2.imwrite(str(output_dir / "contours.png"), contour_image)

    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        cv2.imwrite(
            str(output_dir / "cropped.png"),
            contour_image[y : y + h, x : x + w],
        )

    for i, cnt in enumerate(contours[1:], start=1):
        print(f"Contour {i} area: {cv2.contourArea(cnt):.1f} px^2")

    return contours


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Area conversion (px^2 → mm^2)
# ─────────────────────────────────────────────────────────────────────────────

def contour_area_mm2(contours: list, calibration_path: str | Path) -> list[float]:
    """
    Calculate contour areas in mm^2 using a camera calibration file.

    Args:
        contours:          List of contours from cv2.findContours.
        calibration_path:  Path to a .npz file with keys:
                           'camera_matrix', 'dist_coeffs', 'pixel_to_mm'.

    Returns:
        List of areas in mm^2 (one per contour).
    """
    calibration_path = Path(calibration_path)
    data = np.load(str(calibration_path))

    for key in ("camera_matrix", "dist_coeffs", "pixel_to_mm"):
        if key not in data:
            raise ValueError(f"Missing '{key}' in calibration file: {calibration_path}")

    camera_matrix = data["camera_matrix"]
    dist_coeffs   = data["dist_coeffs"]
    pixel_to_mm   = float(data["pixel_to_mm"])

    areas: list[float] = []
    for cnt in contours:
        undistorted = cv2.undistortPoints(
            cnt.astype(np.float32), camera_matrix, dist_coeffs, P=camera_matrix
        )
        areas.append(cv2.contourArea(undistorted) * pixel_to_mm ** 2)

    return areas


# ─────────────────────────────────────────────────────────────────────────────
# 5.  CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect and measure a dominant-colored object in an image.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Required paths ─────────────────────────────────────────────────────
    parser.add_argument(
        "--input", "-i",
        required=True,
        metavar="IMAGE",
        help="Path to the input image (jpg, png, …).",
    )
    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        metavar="DIR",
        help="Directory where diagnostic images will be saved.",
    )

    # ── Optional calibration ───────────────────────────────────────────────
    parser.add_argument(
        "--calibration", "-c",
        default=None,
        metavar="NPZ",
        help="Path to calibration .npz file. "
             "If omitted, area is reported in px² only.",
    )

    # ── Tuning ────────────────────────────────────────────────────────────
    parser.add_argument("--h-std-factor",   type=float, default=2.0)
    parser.add_argument("--s-std-factor",   type=float, default=1.5)
    parser.add_argument("--v-std-factor",   type=float, default=2.0)
    parser.add_argument(
        "--min-saturation",
        type=int,
        default=60,
        help="Minimum HSV saturation for color sampling (0–255).",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_dir  = Path(args.output_dir)

    # ── Detection ─────────────────────────────────────────────────────────
    contours = object_detection(
        image_path=input_path,
        output_dir=output_dir,
        min_saturation=args.min_saturation,
        h_std_factor=args.h_std_factor,
        s_std_factor=args.s_std_factor,
        v_std_factor=args.v_std_factor,
    )

    measured_contours = contours[1:]  # skip the outermost envelope contour

    if not measured_contours:
        print("\nNo inner contours found — nothing to measure.")
        sys.exit(0)

    # ── Area reporting ─────────────────────────────────────────────────────
    if args.calibration:
        areas = contour_area_mm2(measured_contours, args.calibration)
        unit  = "mm^2"
    else:
        areas = [cv2.contourArea(c) for c in measured_contours]
        unit  = "px^2"
        print("\n[INFO] No calibration file supplied — reporting area in px^2.")

    # ── Label contours on a copy of the image ─────────────────────────────
    image = cv2.imread(str(input_path))
    for i, (cnt, area) in enumerate(zip(measured_contours, areas), start=1):
        x, y, w, h = cv2.boundingRect(cnt)
        label = f"{area:.2f} {unit}"
        cv2.putText(
            image, label,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
        )
        print(f"Contour {i}: {label}")

    labeled_path = output_dir / "labeled.png"
    cv2.imwrite(str(labeled_path), image)
    print(f"\nLabeled image saved → {labeled_path}")

    if areas:
        print(f"Average area: {np.mean(areas):.2f} {unit}")


if __name__ == "__main__":
    main()