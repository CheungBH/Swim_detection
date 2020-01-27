import cv2
import numpy as np
import copy
import math


def get_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def blur(img):
    blur_kernel = (11, 11)
    return cv2.blur(img, blur_kernel)


def enhance(img, kernel="3*3"):
    if kernel == "3*3":
        enhance_kernel = np.array([[0, -1, 0], [0, 5, 0], [0, -1, 0]])
    elif kernel == "5*5":
        enhance_kernel = np.array([
            [-1, -1, -1, -1, -1],
            [-1, 2, 2, 2, -1],
            [-1, 2, 8, 2, -1],
            [-1, 2, 2, 2, -1],
            [-1, -1, -1, -1, -1]]) / 8.0
    else:
        raise ValueError("Wrong size!")
    return cv2.filter2D(img, -1, enhance_kernel)


def get_hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def segment_with_thresh(img):
    lower = np.array([0, 0, 46])
    upper = np.array([180, 255, 255])
    return cv2.inRange(img, lowerb=lower, upperb=upper)


def dilate(img):
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.dilate(img, dilate_kernel)


def erode(img):
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.erode(img, erode_kernel)


def find_contour(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def harris(img):
    image = copy.deepcopy(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(src=gray, blockSize=9, ksize=27, k=0.04)
    a = dst > 0.01 * dst.max()
    image[a] = [0, 0, 255]
    return image

def calcGrayHist(I):
    # 计算灰度直方图
    h, w = I.shape[:2]
    grayHist = np.zeros([256], np.uint64)
    for i in range(h):
        for j in range(w):
            grayHist[I[i][j]] += 1
    return grayHist

def equalHist(img):
    # 灰度图像矩阵的高、宽
    img = get_gray(img)
    h, w = img.shape
    # 第一步：计算灰度直方图
    grayHist = calcGrayHist(img)
    # 第二步：计算累加灰度直方图
    zeroCumuMoment = np.zeros([256], np.uint32)
    for p in range(256):
        if p == 0:
            zeroCumuMoment[p] = grayHist[0]
        else:
            zeroCumuMoment[p] = zeroCumuMoment[p - 1] + grayHist[p]
    # 第三步：根据累加灰度直方图得到输入灰度级和输出灰度级之间的映射关系
    outPut_q = np.zeros([256], np.uint8)
    cofficient = 256.0 / (h * w)
    for p in range(256):
        q = cofficient * float(zeroCumuMoment[p]) - 1
        if q >= 0:
            outPut_q[p] = math.floor(q)
        else:
            outPut_q[p] = 0
    # 第四步：得到直方图均衡化后的图像
    equalHistImage = np.zeros(img.shape, np.uint8)
    for i in range(h):
        for j in range(w):
            equalHistImage[i][j] = outPut_q[img[i][j]]
    return equalHistImage













