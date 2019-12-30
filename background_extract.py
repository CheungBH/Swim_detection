import cv2
# from cut import cut_image
import os
from config import config

cnt = 0
cap = cv2.VideoCapture('Video/Selected/54.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=200, detectShadows=False)
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (540, 360))
    fgmask = fgbg.apply(frame)
    background = fgbg.getBackgroundImage()
    diff = cv2.absdiff(frame, background)
    cv2.imshow('input', frame)
    cv2.moveWindow("input", 0, 0)
    cv2.imshow('background', background)
    cv2.moveWindow("background", 0, 450)
    cv2.imshow('mask',diff)
    cv2.moveWindow("mask", 540, 0)

    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)
    cv2.moveWindow("gray", 1080, 450)

    blur = cv2.blur(diff, config.blur_kernel)

    imageEnhance = cv2.filter2D(blur, -1, config.enhance_kernel)
    cv2.imshow("enhance", imageEnhance)
    cv2.moveWindow("enhance", 540, 450)
    hsv = cv2.cvtColor(imageEnhance, cv2.COLOR_BGR2HSV)

    # thresh_black = cv2.inRange(hsv, lowerb=config.hsv_lower_black, upperb=config.hsv_upper_black)
    # thresh_red = cv2.inRange(hsv, lowerb=config.hsv_lower_red, upperb=config.hsv_upper_red)
    # thresh = cv2.bitwise_and(thresh_black, thresh_red, dst=None, mask=None)

    thresh = cv2.inRange(hsv, lowerb=config.hsv_lower_black, upperb=config.hsv_upper_black)
    cv2.imshow("threshold", thresh)
    cv2.moveWindow("threshold", 1080, 0)

    dilate_kernel = cv2.getStructuringElement(config.dilation_method[0], config.dilation_kernel)
    dilation = cv2.morphologyEx(thresh, config.dilation_method[1], dilate_kernel)
    # cv.imwrite()
    # cut_mask = cut_image(fgmask, top=105)
    # cv.imwrite('new/{}.jpg'.format(cnt), frame)
    cnt += 1
    k = cv2.waitKey(10)&0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()