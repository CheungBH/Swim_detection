import numpy as np


class Utils(object):
    def __init__(self):
        pass

    @staticmethod
    def cal_cosine(vec1, vec2):
        dis1 = Utils.cal_dis(vec1)
        dis2 = Utils.cal_dis(vec2)
        prod = Utils.dot_product(vec1, vec2)
        cos = prod/(dis1 * dis2)
        return cos

    @staticmethod
    def cal_dis(vec):
        sum = 0
        for num in vec:
            sum += num * num
        return sum ** 0.5

    @staticmethod
    def dot_product(v1, v2):
        sum = 0
        for idx in range(len(v1)):
            sum += v1[idx] * v2[idx]
        return sum

    @staticmethod
    def cut_image(img, bottom=0, top=0, left=0, right=0):
        height, width = img.shape[0], img.shape[1]
        return np.asarray(img[top: height - bottom, left: width - right])


if __name__ == '__main__':
    U = Utils()
    vec1, vec2 = [2, 2], [-1, -1]
    print(U.cal_cosine(vec1, vec2))