# -*- coding: utf-8 -*
import cv2
from cv2 import aruco
import numpy as np


# WEBカメラ
cap = cv2.VideoCapture(0)

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters_create()
# CORNER_REFINE_NONE, no refinement. CORNER_REFINE_SUBPIX, do subpixel refinement. CORNER_REFINE_CONTOUR use contour-Points
parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR

cameraMatrix = np.load('params/mtx.npy')
distCoeffs = np.load('params/dist.npy')

cap.set(cv2.CAP_PROP_FPS, 10)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)

ret, frame = cap.read()

# 変換処理ループ
while ret:
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        frame, dictionary, parameters=parameters)
    # print(corners)
    # print(ids)
    # print(rejectedImgPoints)

    aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))

    for i, corner in enumerate(corners):
        points = corner[0].astype(np.int32)
        cv2.polylines(frame, [points], True, (0, 255, 255))
        cv2.putText(frame, str(ids[i][0]), tuple(
            points[0]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

    # rvecs, tvecs, _objPoints =   cv.aruco.estimatePoseSingleMarkers( corners, markerLength, cameraMatrix, distCoeffs[, rvecs[, tvecs[, _objPoints]]] )
    rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(
        corners, 0.05, cameraMatrix, distCoeffs)
    # print('rvec: {}, \ntvec: {}'.format(rvecs, tvecs))
    if ids is not None:
        print('---------------------------')
        for i in range(ids.size):
            print('id: ', ids[i])
            # print('rvec {}, tvec {}'.format(rvecs[i], tvecs[i]))
            print('rvecs[{}] {}'.format(i, rvecs[i]))
            # print('tvecs[{}] {}'.format(i, tvecs[i]))
            aruco.drawAxis(frame, cameraMatrix, distCoeffs,
                           rvecs[i], tvecs[i], 0.1)

    cv2.imshow('org', frame)

    # qキーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 次のフレーム読み込み
    ret, frame = cap.read()

cv2.destroyAllWindows()
cap.release()
