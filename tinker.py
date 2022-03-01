# j=3
# i=3
#
# d={}
# d['d'+str(j)+str(i)]={}
# print(d)
# d['d'+str(j)+str(i)]['d'+str(j+1)+str(i)] = 69
# print(d)

import cv2
import numpy as np

def nothing(x):
    pass

# cv2.namedWindow("Track")

x,y =92,92
while True:
    # cv2.createTrackbar("x", "Track", 0, 100, nothing)
    # cv2.createTrackbar("y", "Track", 0, 100, nothing)
    # x = cv2.getTrackbarPos("x", "Track")
    # y = cv2.getTrackbarPos("y", "Track")
    img = cv2.imread('sample.jpg')
    w = img.shape[0]
    h = img.shape[1]
    roi = img[x:w-x, y:h-y]
    cv2.imshow('Image',roi)
    cv2.waitKey(0)
cv2.destroyAllWindows()