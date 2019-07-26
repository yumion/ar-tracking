# coding:utf-8
import cv2
aruco = cv2.aruco

# ARマーカー生成
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
for i in range(5):
    marker = aruco.drawMarker(dictionary, i + 1, 100)
    cv2.imwrite('ar_marker' + str(i + 1) + '.png', marker)
