# coding: utf-8
import cv2
from cv2 import aruco
import numpy as np


dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

cap = cv2.VideoCapture(0)
# frameサイズ指定
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    # マーカーを検出
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        frame, dictionary)
    if ids is not None:
        print('--------------------------')
        for i in range(ids.size):
            print('id: ', ids[i])
            print('corners: ', corners[i])
    # マーカーを四角で囲む
    aruco.drawDetectedMarkers(frame, corners, ids)
    # 加工済の画像を表示する
    cv2.imshow('Window', frame)
    # qキーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
