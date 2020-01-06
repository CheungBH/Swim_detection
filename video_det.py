import cv2
import os
from config import config
import numpy as np

cnt = 0
save_ls = [175, 333, 556, 888, 1000, 1080]
video_num = 54
img_folder = str(video_num)

cap = cv2.VideoCapture(os.path.join('Video/Selected', '{}.mp4'.format(img_folder)))
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=200, detectShadows=False)
show = False
write = False

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (540, 360))
    cv2.imshow('input', frame)
    fgmask = fgbg.apply(frame)
    background = fgbg.getBackgroundImage()
    diff = cv2.absdiff(frame, background)
    point = diff

    gray = cv2.cvtColor(point, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(src=gray, blockSize=9, ksize=27, k=0.04)
    a = dst > 0.01 * dst.max()
    point[a] = [0, 0, 255]
    cv2.imshow("point", point)

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    blur = cv2.blur(diff, config.blur_kernel)

    imageEnhance = cv2.filter2D(blur, -1, config.enhance_kernel)

    hsv = cv2.cvtColor(imageEnhance, cv2.COLOR_BGR2HSV)

    thresh = cv2.inRange(hsv, lowerb=config.hsv_lower_black, upperb=config.hsv_upper_black)

    dilate_kernel = cv2.getStructuringElement(config.dilation_method[0], config.dilation_kernel)
    dilation = cv2.morphologyEx(thresh, config.dilation_method[1], dilate_kernel)

    if show:
        cv2.imshow('input', frame)
        cv2.moveWindow("input", 0, 0)
        cv2.imshow('background', background)
        cv2.moveWindow("background", 0, 450)
        cv2.imshow('mask', diff)
        cv2.moveWindow("mask", 540, 0)
        cv2.imshow('gray', gray)
        cv2.moveWindow("gray", 1080, 450)
        cv2.imshow("enhance", imageEnhance)
        cv2.moveWindow("enhance", 540, 450)
        cv2.imshow("threshold", thresh)
        cv2.moveWindow("threshold", 1080, 0)

    if write:
        if cnt in save_ls:
            des_path = os.path.join("frame/{}/{}".format(img_folder, cnt))
            os.makedirs(des_path, exist_ok=True)
            cv2.imwrite(os.path.join(des_path, "mask.jpg"), diff)
            cv2.imwrite(os.path.join(des_path, "enhance.jpg"), imageEnhance)
            cv2.imwrite(os.path.join(des_path, "gray.jpg"), gray)

    print("Finish processing frame {}".format(cnt))
    cnt += 1
    k = cv2.waitKey(10)&0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()