import numpy as np
import math
import cv2
import cv2.aruco as aruco

# aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
#
# img = aruco.drawMarker(aruco_dict, 55, 400)
#
# cv2.imshow('frame',img)
# cv2.waitKey(0)


# detection
test_img = cv2.imread('sample.jpg')
gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
# _,thresh = cv2.threshold(gray,220,255,cv2.THRESH_BINARY)
# kernel = np.ones((5,5), np.uint8)
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
# # closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#
aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)

parameters = aruco.DetectorParameters_create()

corners, ids, rej = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
# corners = np.squeeze(corners)
ids = np.squeeze(ids)


test_img = aruco.drawDetectedMarkers(test_img,corners)

# pts = corners.astype(int)
# pts = pts.reshape((-1, 1, 2))
# # pts = np.array([[25, 70], [25, 145],
# #                 [75, 190], [150, 190],
# #                 [200, 145], [200, 70],
# #                 [150, 25], [75, 25]],
# #                np.int32)
#
#
print('ID = ', ids)
print(corners)
#
# detected = cv2.polylines(test_img, [pts],True, (0,255,0),2)

# cv2.imshow('detected', thresh)
# cv2.imshow('detected2', opening)
cv2.imshow('MARKERS', test_img)
cv2.waitKey(0)