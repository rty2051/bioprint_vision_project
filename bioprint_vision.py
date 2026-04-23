import cv2
import numpy as np
import time


def get_hsv_at_cursor():
    while True:
        x, y = pyautogui.position()
        screenshot = pyautogui.screenshot()
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        b, g, r = img[y, x]
        hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        print(f"Cursor at ({x}, {y}) - HSV: {hsv}")
        time.sleep(0.5)


def compute_adaptive_green_bounds(hsv_image, h_std_factor=2.0, s_std_factor=1.5, v_std_factor=2.0):
    """
    Adaptively compute HSV bounds for green objects by:
    1. Doing a loose pre-filter for any plausible green hue (H: 35–95)
    2. Sampling the HSV distribution of those pixels
    3. Returning tight bounds centered on the measured mean ± k*std

    Args:
        hsv_image:      Full HSV image (H in [0,179])
        h_std_factor:   How many std-devs to allow around mean Hue
        s_std_factor:   How many std-devs to allow around mean Saturation
        v_std_factor:   How many std-devs to allow around mean Value

    Returns:
        lower (np.array), upper (np.array)  — both in OpenCV HSV scale
    """
    h, s, v = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]

    # ── Loose pre-filter: green hue band + minimum saturation ──────────────
    loose_green = (h >= 35) & (h <= 95) & (s >= 60)
    if loose_green.sum() < 50:
        print("[WARN] Very few green pixels found in pre-filter. "
              "Falling back to hard-coded defaults.")
        return np.array([60, 140, 52]), np.array([85, 255, 235])

    # ── Sample statistics of confirmed green pixels ────────────────────────
    h_vals = h[loose_green].astype(float)
    s_vals = s[loose_green].astype(float)
    v_vals = v[loose_green].astype(float)

    h_mean, h_std = h_vals.mean(), h_vals.std()
    s_mean, s_std = s_vals.mean(), s_vals.std()
    v_mean, v_std = v_vals.mean(), v_vals.std()

    print(f"[Adaptive HSV] H: {h_mean:.1f} ± {h_std:.1f}  "
          f"S: {s_mean:.1f} ± {s_std:.1f}  "
          f"V: {v_mean:.1f} ± {v_std:.1f}")

    lower = np.array([
        max(0,   int(h_mean - h_std_factor * h_std)),
        max(0,   int(s_mean - s_std_factor * s_std)),
        max(70,   int(v_mean - v_std_factor * v_std)),
    ])
    upper = np.array([
        min(80, int(h_mean + h_std_factor * h_std) + 10),
        255,
        min(200, int(v_mean + v_std_factor * v_std) + 10),
    ])

    print(f"[Adaptive HSV] lower={lower}  upper={upper}")
    return lower, upper


def object_detection(image_path=None):
    if image_path is None:
        image_path = (
            "/home/ryam/Desktop/bioprint_vision_project/"
            "images/better_square_photos/better_cam_1.jpg"
        )

    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    # h, w = image.shape[:2]
    # margin_y = int(0.10 * h)
    # margin_x = int(0.10 * w)
    # cropped = image[margin_y:h - margin_y, margin_x:w - margin_x]
        # image = cropped
        # cv2.imwrite("test_images/green_cropped.png", cropped)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # ── Adaptive thresholding ──────────────────────────────────────────────
    lower_green, upper_green = compute_adaptive_green_bounds(hsv)
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # ── Morphological clean-up ─────────────────────────────────────────────
    # Remove small noise
    kernel_open = np.ones((1, 1), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    # Close small holes in the hydrogel body
    kernel_close = np.ones((11, 11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    final_mask = mask
    cv2.imwrite("test_images/green_object_mask.png", final_mask)

    # ── Contour detection ──────────────────────────────────────────────────
    contours, _ = cv2.findContours(final_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 2)

    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        cropped = contour_image[y:y + h, x:x + w]
        cv2.imwrite("test_images/green_object_cropped.png", cropped)

    cv2.imwrite("test_images/green_object_contours.png", contour_image)

    for i, cnt in enumerate(contours[1:]):
        area = cv2.contourArea(cnt)
        print(f"Contour {i + 1} area: {area:.1f} px")

    return contours


def contour_area_mm2(contours, calibration_path):
    """
    Calculate contour areas in mm² using full calibration data.

    Args:
        contours (list): List of contours from cv2.findContours
        calibration_path (str): Path to calibration .npz file

    Returns:
        list: Areas in mm² for each contour
    """
    data = np.load(calibration_path)

    for key in ('camera_matrix', 'dist_coeffs', 'pixel_to_mm'):
        if key not in data:
            raise ValueError(f"Missing '{key}' in calibration file.")

    camera_matrix = data['camera_matrix']
    dist_coeffs   = data['dist_coeffs']
    pixel_to_mm   = float(data['pixel_to_mm'])

    areas_mm2 = []
    for cnt in contours:
        cnt_f = cnt.astype(np.float32)
        undistorted = cv2.undistortPoints(
            cnt_f, camera_matrix, dist_coeffs, P=camera_matrix
        )
        area_px  = cv2.contourArea(undistorted)
        areas_mm2.append(area_px * (pixel_to_mm ** 2))

    return areas_mm2


if __name__ == "__main__":
    # get_hsv_at_cursor()
    path = "images\\A5C5Scaffold2\\layer_001_001.jpg"
    contours  = object_detection(path)
    areas_mm2 = contour_area_mm2(contours[1:], "calibration/calibration_data.npz")

    for i, area in enumerate(areas_mm2):
        print(f"Contour {i + 1} area: {area:.2f} mm²")

    # Label contours on saved image
    image = cv2.imread("test_images/green_object_contours.png")
    for i, cnt in enumerate(contours[1:]):
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.putText(
            image, f"{areas_mm2[i]:.2f} mm^2",
            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (0, 0, 255), 2
        )
    cv2.imwrite("test_images/green_object_labeled.png", image)

    if areas_mm2:
        print(f"\nAverage area: {np.mean(areas_mm2):.2f} mm²")
