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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

width = 235
height = 370
# 実際の座標を決める
true_coordinates = np.float32(
    [[0., 0.], [width, 0.], [0., height], [width, height]])  # id=1,2,3,4の座標

# 動画を保存
fourcc = cv2.VideoWriter_fourcc(*'XVID')
rec = cv2.VideoWriter('results/trajectory.mp4', fourcc, 15.0, (width, height))

# 軌道plotデータ
x_t = []
y_t = []
fig = plt.figure(figsize=(width // 50, height // 50))

while True:
    ret, frame = cap.read()
    # print(ret)
    # マーカー検出
    corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, dictionary)
    img_marked = aruco.drawDetectedMarkers(frame, corners, ids)
    cv2.imwrite('results/test.png', img_marked)

    if ids is None or ids.size < 4:
        cv2.imshow('window', img_marked)

    else:
        if ids.size == 5 and all(ids <= 5):
            moments = calcMoments(corners, ids)
            marker_coordinates = np.float32(moments[:4])  # 四隅
            trans_mat = cv2.getPerspectiveTransform(
                marker_coordinates, true_coordinates)

            target_pos = moments[4]  # 動かすマーカー
            trans_pos = transPos(trans_mat, target_pos)

            x_t.append(trans_pos[0])
            y_t.append(trans_pos[1])

            img_marked = aruco.drawDetectedMarkers(frame, corners, ids)
            img_trans = cv2.warpPerspective(
                img_marked, trans_mat, (width, height))  # 写像する
        else:
            x_t.append(None)
            y_t.append(None)
            moments = calcMoments(corners, ids)
            marker_coordinates = np.float32(moments[:4])  # 四隅
            trans_mat = cv2.getPerspectiveTransform(
                marker_coordinates, true_coordinates)
            img_trans = cv2.warpPerspective(
                frame, trans_mat, (width, height))
        # 表示
        cv2.imshow('window', cv2.flip(img_trans, 0))  # 上下反転を直す
        rec.write(cv2.flip(img_trans, 0))
        plt.scatter(x_t, y_t)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(0, width)
        plt.ylim(0, height)
        plt.savefig("results/trajectory.png")

    # quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

plt.scatter(x_t, y_t)
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, width)
plt.ylim(0, height)
plt.savefig("results/trajectory.png", dpi=300)
# plt.show()

cv2.destroyAllWindows()
cap.release()
rec.release()
