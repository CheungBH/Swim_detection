import numpy as np
import cv2

frame_shape = (1440, 900)
water_top = 500
blur_kernel = (11, 11)
enhance_kernel = np.array([[0, -1, 0], [0, 5, 0], [0, -1, 0]])

hsv_lower = np.array([0, 0, 46])
hsv_upper = np.array([180, 255, 255])

dilation_method = (cv2.MORPH_CROSS, cv2.MORPH_OPEN)
dilation_kernel = (3, 3)

real_con_len = 400
contour_method = (cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
