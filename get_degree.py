# -*- coding: utf-8 -*-
import cv2
import cv2.aruco as aruco
import numpy as np
import math

arucoMarkerLength = 0.05


class AR():

    def __init__(self, videoPort, cameraMatrix, distortionCoefficients):
        self.cap = cv2.VideoCapture(videoPort)
        self.cameraMatrix = np.load(cameraMatrix)
        self.distortionCoefficients = np.load(distortionCoefficients)
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

    def findARMarker(self):
        self.ret, self.frame = self.cap.read()
        if len(self.frame.shape) == 3:
            self.Height, self.Width, self.channels = self.frame.shape[:3]
        else:
            self.Height, self.Width = self.frame.shape[:2]
            self.channels = 1
        self.halfHeight = int(self.Height / 2)
        self.halfWidth = int(self.Width / 2)
        self.corners, self.ids, self.rejectedImgPoints = aruco.detectMarkers(
            self.frame, self.dictionary)
        # corners[id0,1,2...][][corner0,1,2,3][x,y]
        aruco.drawDetectedMarkers(
            self.frame, self.corners, self.ids, (0, 255, 0))

        return self.ids

    def show(self):
        cv2.imshow("result", self.frame)

    def getARPointAverage(self):
        num = self.getExistMarker()
        if num < 1:
            return
        else:
            square_points = np.reshape(np.array(self.corners), (4 * num, -1))
            G = np.mean(square_points, axis=0)
            cv2.circle(self.frame, (int(G[0]), int(
                G[1])), 10, (255, 255, 255), 5)
            x = self.halfHeight - G[0]
            y = G[1] - self.halfWidth
            return x, y

    # AR2つのそれぞれの座標が欲しい場合
    def getARPoint(self):
        if len(self.corners) >= 2:
            square_points = np.reshape(np.array(self.corners), (4, -1))
            G = np.mean(square_points, axis=0)
            cv2.circle(self.frame, (int(G[0]), int(
                G[1])), 10, (255, 255, 255), 5)
            x0 = self.halfHeight - G[0]
            y0 = G[1] - self.halfWidth
            x1 = self.halfHeight - G[2]
            y1 = G[3] - self.halfWidth
            return (x0, y0, x1, y1)

    def getDistanceAverage(self):
        if len(self.corners) > 0:
            self.rvec, self.tvec, _ = aruco.estimatePoseSingleMarkers(
                self.corners, arucoMarkerLength, self.cameraMatrix, self.distortionCoefficients)
            G = np.mean(self.tvec, axis=0)
            return G[0][2]

            # ARそれぞれの距離が欲しかったらこっち
            # return self.tvec[0][0][2], self.tvec[1][0][2]

    def getDegrees(self):
        if len(self.corners) > 0:
            self.rvec, self.tvec, _ = aruco.estimatePoseSingleMarkers(
                self.corners[0], arucoMarkerLength, self.cameraMatrix, self.distortionCoefficients)  # はじめに検出したマーカーの角度を取得
            self.frame = aruco.drawAxis(
                self.frame, self.cameraMatrix, self.distortionCoefficients, self.rvec, self.tvec, 0.1)
            roll_angle = self.rvec[0][0][0] * 180 / math.pi
            pitch_angle = self.rvec[0][0][1] * 180 / math.pi
            yaw_angle = self.rvec[0][0][2] * 180 / math.pi
            if pitch_angle < 0:
                # ARマーカーを正面から撮影するとy軸の正負がどっちにあるか判断できないっぽくて、角度の符号がおかしくなるので修正
                roll_angle, pitch_angle, yaw_angle = -roll_angle, -pitch_angle, -yaw_angle
            return (roll_angle, pitch_angle, yaw_angle)

    def getExistMarker(self):
        return len(self.corners)

    def release(self):
        self.cap.release()


if __name__ == '__main__':

    myCap = AR(0, 'params/mtx.npy', 'params/dist.npy')
    while True:
        ids = myCap.findARMarker()
        print('detect id:', ids)
        print(myCap.getDegrees())
        myCap.show()
        if cv2.waitKey(1) > 0:
            myCap.release()
            cv2.destroyAllWindows()
            break
