# coding: utf-8
import numpy as np
import cv2
import matplotlib.pyplot as plt

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)  # マーカーの辞書


def calcMoments(corners, ids):
    moments = np.empty((len(corners), 2))
    for i in range(len(corners)):
        index = int(ids[i]) - 1
        moments[index] = np.mean(corners[i][0], axis=0)
    return moments


def transPos(trans_mat, target_pos):
    target_pos = np.append(target_pos, 1)
    target_pos_trans = trans_mat @ target_pos  # 内積
    target_pos_trans = target_pos_trans / target_pos_trans[2]
    return target_pos_trans[:2]


cap = cv2.VideoCapture(0)

width = 220
height = 280
true_coordinates = np.float32(
    [[0., 0.], [width, 0.], [0., height], [width, height]])  # 現実の座標

x_t = []
y_t = []
fig = plt.figure()

while True:
    ret, frame = cap.read()
    # print(ret)
    # マーカー検出
    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, dictionary)
    img_marked = aruco.drawDetectedMarkers(frame, corners, ids)
    cv2.imwrite('test.png', img_marked)

    if ids.all() is not None and ids.size == 5 and all(ids <= 5):
        moments = calcMoments(corners, ids)
        marker_coordinates = np.float32(moments[:4])
        trans_mat = cv2.getPerspectiveTransform(
            marker_coordinates, true_coordinates)

        target_pos = moments[4]
        trans_pos = transPos(trans_mat, target_pos)

        x_t.append(trans_pos[0])
        y_t.append(trans_pos[1])

        img_marked = aruco.drawDetectedMarkers(frame, corners, ids)
        img_trans = cv2.warpPerspective(img_marked, trans_mat, (width, height))
    else:
        x_t.append(None)
        y_t.append(None)
        img_trans = cv2.warpPerspective(
            frame, trans_mat, (width, height))

    # 表示
    cv2.imshow('window', img_trans)
    plt.scatter(x_t, y_t)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig("results/tragectory.png", dpi=300)

    # quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

plt.scatter(x_t, y_t)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("results/tragectory.png", dpi=300)
plt.show()

cap.release()
cv2.destroyAllWindows()
