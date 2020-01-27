import os

data_folder = "../data/2Dfilter/"
video_name = "41"
frame_num = [700, 800]
step = 10

frame_ls = [os.path.join(data_folder, "{}_{}.jpg".format(video_name, cnt)) for cnt in range(frame_num[0], frame_num[1]
                                                                                            + step, step)]
# for img in frame_ls:
#     try:
#         os.remove(img)
#     except FileNotFoundError:
#         continue

