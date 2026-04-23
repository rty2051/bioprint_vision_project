import numpy as np
import cv2

# Number of squares (NOT corners)
cols = 7  # 6 inner corners → 7 squares
rows = 5  # 4 inner corners → 5 squares

square_size = 100  # pixels per square

# Create checkerboard
board = np.zeros((rows * square_size, cols * square_size), dtype=np.uint8)

for i in range(rows):
    for j in range(cols):
        if (i + j) % 2 == 0:
            cv2.rectangle(
                board,
                (j * square_size, i * square_size),
                ((j + 1) * square_size, (i + 1) * square_size),
                255,
                -1
            )

cv2.imwrite("test_images//chessboard_6x4.png", board)