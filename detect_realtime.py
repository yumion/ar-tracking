# coding: utf-8
import numpy as np
import cv2
from cv2 import aruco
import matplotlib.pyplot as plt


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


dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)  # マーカーの辞書

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# カメラの内部パラメータ
cameraMatrix = np.load('params/mtx.npy')
distCoeffs = np.load('params/dist.npy')

# 実際の座標を決める
width = 235
height = 370
true_coordinates = np.float32(
    [[0., 0.], [width, 0.], [0., height], [width, height]])  # id=1,2,3,4の座標

# 動画を保存
fourcc = cv2.VideoWriter_fourcc(*'XVID')
rec = cv2.VideoWriter('results/trajectory.mp4', fourcc, 15.0, (width, height))

# 軌道plotデータ
x_t = []  # 座標x
y_t = []  # 座標y
u_t = []  # 向きx成分
v_t = []  # 向きy成分
fig = plt.figure(figsize=(width // 50, height // 50))

trans_mat = None
while True:
    ret, frame = cap.read()
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

            # 向きを検出
            target_id = 5
            for i in range(ids.size):
                if target_id == ids.ravel()[i]:
                    target_corner = corners[i]
            rvec, tvec, _objPoints = aruco.estimatePoseSingleMarkers(
                target_corner, 0.05, cameraMatrix, distCoeffs)
            # 3軸の回転角度を取得
            roll = rvec[0][0][0]
            pitch = rvec[0][0][1]
            yaw = rvec[0][0][2]
            print('roll: {}, pitch: {}, yaw: {}'.format(roll, pitch, yaw))

            # 軌道を保存
            x_t.append(trans_pos[0])
            y_t.append(trans_pos[1])
            u_t.append(20 * np.cos(np.abs(yaw)))  # 大きさ1とした時のx成分
            v_t.append(20 * np.sin(np.abs(yaw)))  # 大きさ1とした時のy成分

            img_marked = aruco.drawDetectedMarkers(
                frame, corners, ids)  # マーカーの枠を描画
            img_marked = aruco.drawAxis(frame, cameraMatrix, distCoeffs,
                                        rvec, tvec, 0.1)  # 軸を描画
            img_trans = cv2.warpPerspective(
                img_marked, trans_mat, (width, height))  # 射影変換
        else:
            # x_t.append(None)
            # y_t.append(None)
            # u_t.append(None)
            # v_t.append(None)
            # 射影変換
            if trans_mat is None:
                moments = calcMoments(corners, ids)
                marker_coordinates = np.float32(moments[:4])  # 四隅
                trans_mat = cv2.getPerspectiveTransform(
                    marker_coordinates, true_coordinates)
            img_trans = cv2.warpPerspective(
                frame, trans_mat, (width, height))
        # 表示
        cv2.imshow('window', cv2.flip(img_trans, 0))  # 上下反転を直す
        rec.write(cv2.flip(img_trans, 0))
        # plot
        plt.scatter(x_t, y_t, c='blue')
        plt.quiver(x_t, y_t, u_t, v_t, angles='xy', scale_units='xy',
                   scale=10, color='blue', alpha=0.5, width=0.01)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(0, width)
        plt.ylim(0, height)
        plt.savefig("results/trajectory.png")

    # quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

plt.scatter(x_t, y_t, c='blue')
plt.quiver(x_t, y_t, u_t, v_t, angles='xy', scale_units='xy',
           scale=1, color='blue', alpha=0.5, width=0.005)
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(0, width)
plt.ylim(0, height)
plt.savefig("results/trajectory.png", dpi=300)
# plt.show()

cv2.destroyAllWindows()
cap.release()
rec.release()
