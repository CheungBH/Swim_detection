import cv2
from utils.utils import Utils
import numpy as np

water_top = 105
standard_frame = cv2.imread("img/origin_side.jpg")


def cut_image(img, bottom=0, top=0, left=0, right=0):
    height, width = img.shape[0], img.shape[1]
    return np.asarray(img[top: height - bottom, left: width - right])


def detect_people(frame):
    frame = cv2.resize(frame, (standard_frame.shape[1], standard_frame.shape[0]))
    diff = cv2.absdiff(frame, standard_frame)
    cut_diff = cut_image(diff, top=water_top)
    blur = cv2.blur(cut_diff, (7, 7))
    enhance_kernel = np.array([[0, -1, 0], [0, 5, 0], [0, -1, 0]])
    imageEnhance = cv2.filter2D(blur, -1, enhance_kernel)
    hsv = cv2.cvtColor(imageEnhance, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, 46])
    upper = np.array([180, 255, 255])
    thresh = cv2.inRange(hsv, lowerb=lower, upperb=upper)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilation = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, dilate_kernel)
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    stored = [idx for idx in range(len(contours)) if len(contours[idx]) > 80]
    real_con = [contours[i] for i in stored]
    rect = []
    for c in real_con:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y + water_top), (x + w, y + water_top + h), (0, 255, 0), 2)
        rect.append([x, y + water_top, x + w, y + water_top + h])
    # cv2.imshow("detection", frame)
    # cv2.waitKey(1000)
    return rect, frame


if __name__ == '__main__':
    img_path = "img/multiple.jpg"
    img = cv2.imread(img_path)
    bbox, frm = detect_people(img)
    print(bbox)
    cv2.imshow("detection", frm)
    cv2.waitKey(100000)
