
import numpy as np
import cv2
aruco = cv2.aruco


WINDOW_NAME = "window"
ORG_FILE_NAME = "data/IMG_8372.MOV"  # gifだとエラー
NEW_FILE_NAME = "results/new.mp4"


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


org = cv2.VideoCapture(ORG_FILE_NAME)
end_flag, original_img = org.read()
width = 1000
height = 1000
fourcc = cv2.VideoWriter_fourcc(*'XVID')
rec = cv2.VideoWriter(NEW_FILE_NAME, fourcc, 60.0, (width, height))
cv2.namedWindow(WINDOW_NAME)

# 1フレーム目を検出
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
# corners, ids, rejectedImgPoints = aruco.detectMarkers(original_img, dictionary)
# img_marked = aruco.drawDetectedMarkers(original_img, corners, ids)
# cv2.imwrite('results/detect.png', img_marked)
# print(all(ids > 5))


# while True:
#     corners, ids, rejectedImgPoints = aruco.detectMarkers(
#         original_img, dictionary)
#     if ids.all() is not None and ids.size > 4:
#         break
#     end_flag, original_img = org.read()
# moments = calcMoments(corners, ids)

# # 座標変換
# marker_coordinates = np.float32(moments[:4])
true_coordinates = np.float32(
    [[0., 0.], [width, 0.], [0., height], [width, height]])
# trans_mat = cv2.getPerspectiveTransform(marker_coordinates, true_coordinates)
# img_trans = cv2.warpPerspective(original_img, trans_mat, (width, height))
# cv2.imwrite('results/trans.png', img_trans)


x_t = []
y_t = []
while end_flag is True:
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        original_img, dictionary)
    if ids.all() is not None and ids.size == 5 and all(ids <= 5):
        moments = calcMoments(corners, ids)
        marker_coordinates = np.float32(moments[:4])
        trans_mat = cv2.getPerspectiveTransform(
            marker_coordinates, true_coordinates)

        target_pos = moments[4]
        trans_pos = transPos(trans_mat, target_pos)

        x_t.append(trans_pos[0])
        y_t.append(trans_pos[1])

        img_marked = aruco.drawDetectedMarkers(original_img, corners, ids)
        img_trans = cv2.warpPerspective(img_marked, trans_mat, (width, height))
    else:
        x_t.append(None)
        y_t.append(None)
        img_trans = cv2.warpPerspective(
            original_img, trans_mat, (width, height))

    cv2.imshow(WINDOW_NAME, img_trans)
    rec.write(img_trans)

    end_flag, original_img = org.read()

cv2.destroyAllWindows()
org.release()
rec.release()

import matplotlib.pyplot as plt
fig = plt.figure()
plt.scatter(x_t, y_t)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("results/tragectory.png", dpi=300)
plt.show()
