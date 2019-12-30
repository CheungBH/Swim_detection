import cv2
from utils.utils import Utils
import os
from config import config

water_top = config.water_top
write = True
show = False

folder_name = "new1"
main_path = os.path.join("processor", folder_name)
files = [name for name in os.listdir(main_path) if os.path.isdir(os.path.join(main_path, name))]
origin_frame = "processor/{}/origin.jpg".format(folder_name)
standard_frame = cv2.imread(origin_frame)
standard_frame = cv2.resize(standard_frame, config.frame_shape)

# files = [34]

standard_frame = cv2.resize(standard_frame, config.frame_shape)
for img_num in files:
    print('Processing img {}'.format(img_num))
    detect_img = "processor/{0}/{1}/{1}.jpg".format(folder_name, img_num)

    frame = cv2.imread(detect_img)
    frame = cv2.resize(frame, (standard_frame.shape[1], standard_frame.shape[0]))
    diff = cv2.absdiff(frame, standard_frame)

    cut_diff = Utils.cut_image(diff, top=water_top)
    cut_img = Utils.cut_image(frame, top=water_top)
    con_frame = cut_img
    real_con_frame = con_frame

    blur = cv2.blur(cut_diff, config.blur_kernel)

    imageEnhance = cv2.filter2D(blur, -1, config.enhance_kernel)

    hsv = cv2.cvtColor(imageEnhance, cv2.COLOR_BGR2HSV)

    # thresh_black = cv2.inRange(hsv, lowerb=config.hsv_lower_black, upperb=config.hsv_upper_black)
    # thresh_red = cv2.inRange(hsv, lowerb=config.hsv_lower_red, upperb=config.hsv_upper_red)
    # thresh = cv2.bitwise_and(thresh_black, thresh_red, dst=None, mask=None)

    thresh = cv2.inRange(hsv, lowerb=config.hsv_lower_black, upperb=config.hsv_upper_black)
    dilate_kernel = cv2.getStructuringElement(config.dilation_method[0], config.dilation_kernel)
    dilation = cv2.morphologyEx(thresh, config.dilation_method[1], dilate_kernel)

    contours, hierarchy = cv2.findContours(dilation, config.contour_method[0], config.contour_method[1])
    stored = [idx for idx in range(len(contours)) if len(contours[idx]) > config.real_con_len]
    real_con = [contours[i] for i in stored]

    # cv2.imshow("img", frame)
    # cv2.waitKey(0)
    for idx, c in enumerate(real_con):
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y + water_top), (x + w, y + water_top + h), (0, 255, 0), 2)
        # cv2.imwrite("square_{}".format(idx), Utils.cut_image(frame, ))

    cv2.drawContours(real_con_frame, real_con, -1, (0, 0, 255), 3, lineType=cv2.LINE_AA)
    cv2.drawContours(con_frame, real_con, -1, (0, 0, 255), 3, lineType=cv2.LINE_AA)

    if show:
        cv2.imshow("diff", diff)
        cv2.moveWindow("diff", 0, 0)

        cv2.imshow("cut", cut_diff)
        cv2.moveWindow("cut", 640, 0)

        cv2.imshow("blur", blur)
        cv2.moveWindow("blur", 1280, 0)

        cv2.imshow("laplancian", imageEnhance)
        cv2.moveWindow("laplancian", 640, 350)

        cv2.imshow("thresh", thresh)
        cv2.moveWindow("thresh", 0, 700)

        cv2.imshow("dilation", dilation)
        cv2.moveWindow("dilation", 640, 700)

        cv2.imshow("contour", con_frame)

        cv2.imshow("detection", frame)
        cv2.moveWindow("detection", 1280, 600)

        cv2.waitKey(0)

    if write:
        out_folder = "processor/{}/{}".format(folder_name, img_num)
        os.makedirs(out_folder, exist_ok=True)
        cv2.imwrite("{}/1_diff.jpg".format(out_folder), diff)
        cv2.imwrite("{}/2_cut_diff.jpg".format(out_folder), cut_diff)
        cv2.imwrite("{}/3_blur.jpg".format(out_folder), blur)
        cv2.imwrite("{}/4_enhance.jpg".format(out_folder), imageEnhance)
        # cv2.imwrite("{}/5_thresh_black.jpg".format(out_folder), thresh_black)
        # cv2.imwrite("{}/5_thresh_red.jpg".format(out_folder), thresh_red)
        cv2.imwrite("{}/5_thresh.jpg".format(out_folder), thresh)
        cv2.imwrite("{}/6_dilation.jpg".format(out_folder), dilation)
        cv2.imwrite("{}/7_contour.jpg".format(out_folder), con_frame)
        cv2.imwrite("{}/7_real_contour.jpg".format(out_folder), real_con_frame)
        cv2.imwrite("{}/8_result.jpg".format(out_folder), frame)



