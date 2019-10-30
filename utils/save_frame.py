import cv2

video_path = '46.mp4'
cnt = 0
cap = cv2.VideoCapture(video_path)

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("{}.jpg".format(cnt), frame)
        cnt += 1
    else:
        break