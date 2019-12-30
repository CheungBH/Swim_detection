import numpy as np
import cv2

frame_shape = (1080, 720)
water_top = 480
blur_kernel = (11, 11)
enhance_kernel = np.array([[0, -1, 0], [0, 5, 0], [0, -1, 0]])

hsv_lower_black = np.array([0, 0, 46])
hsv_upper_black = np.array([180, 255, 255])

hsv_lower_red = np.array([156, 43, 46])
hsv_upper_red = np.array([180, 255, 255])

dilation_method = (cv2.MORPH_CROSS, cv2.MORPH_OPEN)
dilation_kernel = (3, 3)

real_con_len = 400
contour_method = (cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
