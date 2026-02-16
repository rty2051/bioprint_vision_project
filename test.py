import cv2

img = cv2.imread("/home/ryam/Desktop/bioprint_vision_project/images/no_back_orange.jpg")
h, w = img.shape[:2]
offset_x, offset_y = 0, 0
dragging = False
start_x, start_y = 0, 0

display = img.copy()

def mouse_event(event, x, y, flags, param):
    global offset_x, offset_y, dragging, start_x, start_y, display

    if event == cv2.EVENT_LBUTTONDOWN:
        dragging = True
        start_x, start_y = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging:
            dx = x - start_x
            dy = y - start_y
            offset_x += dx
            offset_y += dy
            start_x, start_y = x, y

        update_display(x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False

def update_display(x, y):
    global display

    display = img.copy()

    # Clamp coordinates
    if 0 <= x < w and 0 <= y < h:
        b, g, r = img[y, x]
        cv2.putText(display, f"x={x}, y={y}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(display, f"BGR=({b},{g},{r})", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    # Apply translation (pan)
    M = [[1, 0, offset_x],
         [0, 1, offset_y]]

    display = cv2.warpAffine(display, 
                             cv2.UMat(M).get(), 
                             (w, h),
                             borderMode=cv2.BORDER_CONSTANT)

cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Image", mouse_event)

while True:
    cv2.imshow("Image", display)
    if cv2.waitKey(16) & 0xFF == 27:  # ESC
        break

cv2.destroyAllWindows()
