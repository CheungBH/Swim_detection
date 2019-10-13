import cv2
from utils.utils import Utils
import os
import numpy as np

water_top = 105
write = True

img_num = 1
detect_img = "suspect/{}.jpg".format(img_num)
origin_frame = "img/origin.jpg"
standard_frame = cv2.imread(origin_frame)

frame = cv2.imread(detect_img)
frame = cv2.resize(frame, (standard_frame.shape[1], standard_frame.shape[0]))
diff = cv2.absdiff(frame, standard_frame)
cv2.imshow("diff", diff)

cut_diff = Utils.cut_image(diff, top=water_top)
cv2.imshow("cut", cut_diff)

enhance_kernel = np.array([[0, -1, 0], [0, 5, 0], [0, -1, 0]])
imageEnhance = cv2.filter2D(cut_diff, -1, enhance_kernel)
cv2.imshow("laplancian", imageEnhance)

hsv = cv2.cvtColor(imageEnhance, cv2.COLOR_BGR2HSV)

lower = np.array([0, 0, 46])
upper = np.array([180, 255, 255])

thresh = cv2.inRange(hsv, lowerb=lower, upperb=upper)
cv2.imshow("thresh", thresh)

open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, open_kernel)
cv2.imshow("opening", opening)

contours, hierarchy = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
stored = [idx for idx in range(len(contours)) if len(contours[idx]) > 80]
real_con = [contours[i] for i in stored]

for c in real_con:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(frame, (x, y + water_top), (x + w, y + water_top + h), (0, 255, 0), 2)

if write:
    out_folder = "mid/{}".format(img_num)
    os.makedirs(out_folder, exist_ok=True)
    cv2.imwrite("{}/diff.jpg".format(out_folder), diff)
    cv2.imwrite("{}/cut_diff.jpg".format(out_folder), cut_diff)
    cv2.imwrite("{}/enhance.jpg".format(out_folder), imageEnhance)
    cv2.imwrite("{}/thresh.jpg".format(out_folder), thresh)
    cv2.imwrite("{}/opening.jpg".format(out_folder), opening)
    cv2.imwrite("{}/result.jpg".format(out_folder), frame)

cv2.imshow("detection", frame)
cv2.waitKey(0)
