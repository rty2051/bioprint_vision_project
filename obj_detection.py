import cv2
import numpy as np 
import pyautogui
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
  TEST_IMAGE = "images\\better_square_photos\\better_cam_1.jpg"

  # Load image
  image = cv2.imread(TEST_IMAGE)

  # Convert to HSV
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

  # Lower and upper bounds for green in HSV
  lower_green = np.array([60, 110, 52])   # H, S, V
  upper_green = np.array([85, 255, 235])

  mask = cv2.inRange(hsv, lower_green, upper_green)

  # Remove small noise
  kernel = np.ones((9, 9), np.uint8)
  mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
  # Close small holes
  mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

  # Initialize GrabCut with the mask
  bgdModel = np.zeros((1,65), np.float64)
  fgdModel = np.zeros((1,65), np.float64)
  mask2 = np.where(mask==255, 1, 0).astype('uint8')

  cv2.grabCut(image, mask2, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
  final_mask = np.where((mask2==1) | (mask2==3), 255, 0).astype('uint8')

  cv2.imwrite("test_images/green_object_mask.png", final_mask)

  # Find contours in the final mask
  contours, _ = cv2.findContours(final_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

  # Draw contours on a copy of the original image
  contour_image = image.copy()
  cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 2)
  cv2.imwrite("test_images/green_object_contours.png", contour_image)

if __name__ == "__main__":
    print("Press Ctrl+C to stop.")
    get_hsv_at_cursor()
    # object_detection()