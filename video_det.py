import os
import cv2
import numpy as np
from utils.utils import Utils
import sys

video_path = sys.argv[1]
cap = cv2.VideoCapture(os.path.join("img", video_path))
water_top = 105
standard_frame = cv2.imread("img/origin.jpg")
os.makedirs("frame", exist_ok=True)
cnt = 0

while True:
    ret, origin_frame = cap.read()
    if ret:
        frame = cv2.resize(origin_frame, (standard_frame.shape[1], standard_frame.shape[0]))
        diff = cv2.absdiff(frame, standard_frame)
        cut_diff = Utils.cut_image(diff, top=water_top)
        blur = cv2.blur(cut_diff, (7, 7))
        enhance_kernel = np.array([[0, -1, 0], [0, 5, 0], [0, -1, 0]])
        imageEnhance = cv2.filter2D(blur, -1, enhance_kernel)
        hsv = cv2.cvtColor(imageEnhance, cv2.COLOR_BGR2HSV)
        thresh = cv2.inRange(hsv, lowerb=np.array([0, 0, 46]), upperb=np.array([180, 255, 255]))
        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        dilation = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, dilate_kernel)
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        stored = [idx for idx in range(len(contours)) if len(contours[idx]) > 80]
        rects = []
        real_con = [contours[i] for i in stored]
        for c in real_con:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y + water_top), (x + w, y + water_top + h), (0, 255, 0), 2)
            rects = [x, y, x + w, y + water_top + h]
        cv2.imshow("detection", frame)
        #cv2.imwrite(os.path.join("frame", "{}.jpg".format(cnt)), frame)
        if "multiple" in video_path:
            time = 2
        else:
            time = 30
        cv2.waitKey(time)
        print("Frame: {}".format(cnt))
        cnt += 1
    else:
        break
