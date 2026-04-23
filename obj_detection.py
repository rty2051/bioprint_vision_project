import cv2
import numpy as np 
# import pyautogui
import cv2
import time

def get_hsv_at_cursor():
    while True:
        # Get current mouse position
        x, y = pyautogui.position()
        
        # Take a screenshot
        screenshot = pyautogui.screenshot()
        
        # Convert the screenshot to a numpy array (OpenCV uses BGR)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Get BGR value at cursor
        b, g, r = img[y, x]
        
        # Convert BGR to HSV
        hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        
        print(f"Cursor at ({x}, {y}) - HSV: {hsv}")
        
        time.sleep(0.5)  # update every 0.5 seconds

def object_detection():
  # TEST_IMAGE = "images\\better_square_photos\\better_cam_1.jpg"
  TEST_IMAGE = "images\A5C5UniformScaffold3\DETECTED2.jpg"

  # Load image
  image = cv2.imread(TEST_IMAGE)

  # Convert to HSV
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

  # Lower and upper bounds for green in HSV
  lower_green = np.array([60, 140, 52])   # H, S, V
  upper_green = np.array([85, 255, 235])

  mask = cv2.inRange(hsv, lower_green, upper_green)

  # Remove small noise
  kernel = np.ones((1, 1), np.uint8)
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
  # Close small holes
  kernel = np.ones((11, 11), np.uint8)
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

  # Initialize GrabCut with the mask
  # bgdModel = np.zeros((1,65), np.float64)
  # fgdModel = np.zeros((1,65), np.float64)
  # mask2 = np.where(mask==255, 1, 0).astype('uint8')
  # # mask2 = np.full(mask.shape, cv2.GC_PR_BGD, dtype=np.uint8)
  # # mask2[mask == 255] = cv2.GC_FGD

  # cv2.grabCut(image, mask2, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
  # final_mask = np.where((mask2==1) | (mask2==3), 255, 0).astype('uint8')
  final_mask = mask
  cv2.imwrite("test_images/green_object_mask.png", final_mask)

  # Find contours in the final mask
  contours, _ = cv2.findContours(final_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  # Draw contours on a copy of the original image
  contour_image = image.copy()
  cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 5)

  # Crop to contour [0]
  x, y, w, h = cv2.boundingRect(contours[0])
  cropped = contour_image[y:y+h, x:x+w]
  cv2.imwrite("test_images/green_object_cropped.png", cropped)
  cv2.imwrite("test_images/green_object_contours.png", contour_image)

  # Compute the area of each contour
  for i, cnt in enumerate(contours[1:]):
    area = cv2.contourArea(cnt)
    print(f"Contour {i+1} area:", area, "px")

  return contours

def contour_area_mm2(contours, calibration_path):
    """
    Calculate contour areas in mm^2 using full calibration data.

    Args:
        contours (list): List of contours (from cv2.findContours)
        calibration_path (str): Path to calibration .npz file

    Returns:
        list: Areas in mm^2 for each contour
    """

    # === LOAD CALIBRATION ===
    data = np.load(calibration_path)

    required_keys = ['camera_matrix', 'dist_coeffs', 'pixel_to_mm']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Missing '{key}' in calibration file.")

    camera_matrix = data['camera_matrix']
    dist_coeffs = data['dist_coeffs']
    pixel_to_mm = float(data['pixel_to_mm'])  # ensure scalar

    areas_mm2 = []

    for cnt in contours:
        # === UNDISTORT CONTOUR POINTS ===
        cnt = cnt.astype(np.float32)

        undistorted = cv2.undistortPoints(
            cnt,
            camera_matrix,
            dist_coeffs,
            P=camera_matrix  # keep in pixel coordinate system
        )

        # === AREA IN PIXELS ===
        area_px = cv2.contourArea(undistorted)

        # === CONVERT TO mm^2 ===
        area_mm2 = area_px * (pixel_to_mm ** 2)

        areas_mm2.append(area_mm2)

    return areas_mm2

if __name__ == "__main__":
    print("Press Ctrl+C to stop.")
    # get_hsv_at_cursor()
    contours = object_detection()
    areas_mm2 = contour_area_mm2(contours[1:], "calibration/calibration_data.npz")
    for i, area in enumerate(areas_mm2):
        print(f"Contour {i + 1} area in mm^2: {area:.2f}")

    # Label contours with area
    image = cv2.imread("/home/ryam/Desktop/bioprint_vision_project/images/better_square_photos/better_cam_1.jpg")
    for i, cnt in enumerate(contours[1:]):
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.putText(image, f"{areas_mm2[i]:.2f} mm^2", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite("test_images/green_object_labeled.png", image)

    # Print average area
    if areas_mm2:
        avg_area = np.mean(areas_mm2)
        print(f"Average area in mm^2: {avg_area:.2f}")
