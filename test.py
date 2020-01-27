import cv2
import os
from config import config
import numpy as np
import copy
from utils.opencv_src import *


step = 10
show = False
write = True


class DigitalImageProcess(object):
    def __init__(self):
        self.dest_folder = "data/2Dfilter"
        os.makedirs(self.dest_folder, exist_ok=True)
        self.src_folder = "Video/Selected/train"

    def process_video(self, name):
        cnt = 1
        cap = cv2.VideoCapture(os.path.join(self.src_folder, name))
        fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=200, detectShadows=False)
        print("Begin processing {}".format(name))

        while True:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (540, 360))
                if show:
                    cv2.imshow('input', frame)
                    cv2.moveWindow("input", 0, 0)

                fgmask = fgbg.apply(frame)
                background = fgbg.getBackgroundImage()
                img = cv2.absdiff(frame, background)
                if show:
                    cv2.imshow("background", background)
                    cv2.moveWindow("background", 0, 450)
                    cv2.imshow("diff", img)
                    cv2.moveWindow("diff", 540, 0)

                img = enhance(img)
                if cnt % step == 0:
                    if write:
                        cv2.imwrite(os.path.join(os.path.join(self.dest_folder, "{}_{}.jpg".format(name[:-4], cnt))), img)

                if show:
                    cv2.imshow("enhance", img)
                    cv2.moveWindow("enhance", 540, 450)

                if cnt % step == 0:
                    print("frame {}".format(cnt))
                cnt += 1
                k = cv2.waitKey(10) & 0xff
                if k == 27:
                    break
            else:
                cap.release()
                cv2.destroyAllWindows()
                print("Finish processing {}\n\n".format(name))
                break

    def process_video_folder(self):
        video_name_ls = [video_name for video_name in os.listdir(self.src_folder)]
        for idx, path in enumerate(video_name_ls):
            self.process_video(video_name_ls[idx])


if __name__ == '__main__':
    DIP = DigitalImageProcess()
    DIP.process_video_folder()