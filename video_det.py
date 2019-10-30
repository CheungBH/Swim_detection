import os
import cv2
import numpy as np
from utils.utils import Utils
from config import config

main_folder = "img/drown1"
video_path = os.path.join(main_folder, "drown.mp4")
cap = cv2.VideoCapture(video_path)
water_top = config.water_top
standard_frame = cv2.imread(os.path.join(main_folder, "origin.jpg"))
standard_frame = cv2.resize(standard_frame, config.frame_shape)
os.makedirs(os.path.join(main_folder, "result"), exist_ok=True)
os.makedirs(os.path.join(main_folder, "origin_frame"), exist_ok=True)
cnt = 0

while True:
    ret, origin_frame = cap.read()
    if ret:
        frame = cv2.resize(origin_frame, (standard_frame.shape[1], standard_frame.shape[0]))
        #cv2.imwrite(os.path.join(main_folder, "origin_frame", "{}.jpg".format(cnt)), frame)
        diff = cv2.absdiff(frame, standard_frame)
        cut_diff = Utils.cut_image(diff, top=water_top)
        blur = cv2.blur(cut_diff, config.blur_kernel)
        imageEnhance = cv2.filter2D(blur, -1, config.enhance_kernel)
        hsv = cv2.cvtColor(imageEnhance, cv2.COLOR_BGR2HSV)
        thresh = cv2.inRange(hsv, lowerb=config.hsv_lower, upperb=config.hsv_upper)
        dilate_kernel = cv2.getStructuringElement(config.dilation_method[0], config.dilation_kernel)
        dilation = cv2.morphologyEx(thresh, config.dilation_method[1], dilate_kernel)
        contours, hierarchy = cv2.findContours(dilation, config.contour_method[0], config.contour_method[1])
        stored = [idx for idx in range(len(contours)) if len(contours[idx]) > config.real_con_len]
        rects = []
        real_con = [contours[i] for i in stored]
        for c in real_con:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y + water_top), (x + w, y + water_top + h), (0, 255, 0), 2)
            rects = [x, y, x + w, y + water_top + h]
        cv2.imshow("detection", frame)
        #cv2.imwrite(os.path.join(main_folder, "result", "{}.jpg".format(cnt)), frame)
        cv2.waitKey(2)
        print("Frame: {}".format(cnt))
        cnt += 1
    else:
        break
