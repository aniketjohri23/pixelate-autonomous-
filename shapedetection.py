import cv2
import numpy as np
import math


def shape_detection(roi,h,s,v):
    # roi = cv2.resize(roi, (250, 250))
    lower = np.array([h - 3, s - 3, v - 3])
    upper = np.array([h, s, v])
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(hsv, lower, upper)


    triangle_corners = np.zeros((3, 2), int)
    triangle_direction = np.zeros(2,int) #(y,x)

    bgr_onlyblue = cv2.bitwise_and(bgr,bgr, mask=mask)
    gray_onlyblue = cv2.cvtColor(bgr_onlyblue, cv2.COLOR_BGR2GRAY)
    _, threshold_binary = cv2.threshold(gray_onlyblue, 10, 255, cv2.THRESH_BINARY)
    cv2.imshow("thresh",mask)
    contours, _ = cv2.findContours(threshold_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area= np.zeros(2)
    w = roi.shape[0]
    h = roi.shape[1]
    # for cnt in contours:
    #     approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
    #     cv2.drawContours(roi, [approx], 0, (0), 5)
    #     x = approx.ravel()[0]
    #     y = approx.ravel()[1]
    #     if len(approx) == 3:
    #         cv2.putText(roi, "Triangle", (x, y+20), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0))
    #     elif len(approx) == 4:
    #         cv2.putText(roi, "Rectangle", (x, y+20), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0))
    #     elif len(approx) == 5:
    #         cv2.putText(roi, "Pentagon", (x, y+20), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0))
    #     elif 6 < len(approx) < 15:
    #         cv2.putText(roi, "Ellipse", (x, y+20), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0))
    #     else:
    #         cv2.putText(roi, "Circle", (x, y+20), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0))

    for cnt in contours:
        # print(cnt,'dhappa')
        approx = cv2.approxPolyDP(cnt,  0.01* cv2.arcLength(cnt, True), True)
        cv2.drawContours(roi, [approx], 0, (0), 5)

        for i in range(3):
            triangle_corners[i][0] = approx.ravel()[2*i]
            triangle_corners[i][1] = approx.ravel()[2 * i+1]
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        # print(x,y)
        cv2.circle(roi,(x,y),3,255,-1)

        if (area[0]<cv2.contourArea(cnt) and cv2.contourArea(cnt)<w*h-1000):
            area[0] = cv2.contourArea(cnt)
            area[1] = len(approx)

        # print(approx)

    if area[1] == 3:
        print('TRIANGLE')
        # print(triangle_corners)
        # triangle direction detection
        # triangle_corners
        diff = np.zeros(6,int)
        diff[0] = abs(triangle_corners[2][0] - triangle_corners[1][0])#0
        diff[1] = abs(triangle_corners[0][0] - triangle_corners[2][0])#1
        diff[2] = abs(triangle_corners[0][0] - triangle_corners[1][0])#2

        diff[3] = abs(triangle_corners[2][1] - triangle_corners[1][1])#0
        diff[4] = abs(triangle_corners[0][1] - triangle_corners[2][1])#1
        diff[5] = abs(triangle_corners[0][1] - triangle_corners[1][1])#2

        indexofmin = np.where(diff == np.amin(diff))

        print(indexofmin[0][0],diff[indexofmin[0][0]])
        print(diff)

        if(indexofmin[0][0]<=2 and indexofmin[0][0]>=0):#left,right
            triangle_direction[0] = 0   #(y,x)
            if indexofmin[0][0] == 0:
                triangle_direction[1] = np.sign(
                    triangle_corners[0][0] - (triangle_corners[2][0] + triangle_corners[1][0]) / 2)#(x,y)

            if indexofmin[0][0] == 1:
                triangle_direction[1] = np.sign(
                    triangle_corners[1][0] - (triangle_corners[2][0] + triangle_corners[1][0]) / 2)#(x,y)

            if indexofmin[0][0] == 2:
                triangle_direction[1] = np.sign(
                    triangle_corners[2][0] - (triangle_corners[0][0] + triangle_corners[1][0]) / 2)#(x,y)

        if (indexofmin[0][0] <= 5 and indexofmin[0][0] >= 3):#up,down
            triangle_direction[1] = 0  # (y,x)
            if indexofmin[0][0] == 3:
                triangle_direction[0] = np.sign(
                    triangle_corners[0][1] - (triangle_corners[2][1] + triangle_corners[1][1]) / 2)  # (x,y)

            if indexofmin[0][0] == 4:
                triangle_direction[0] = np.sign(
                    triangle_corners[1][1] - (triangle_corners[2][1] + triangle_corners[1][1]) / 2)  # (x,y)

            if indexofmin[0][0] == 5:
                triangle_direction[0] = np.sign(
                    triangle_corners[2][1] - (triangle_corners[0][1] + triangle_corners[1][1]) / 2)  # (x,y)

        print(triangle_direction)




    elif area[1] == 4:
        print('SQUARE')
        triangle_direction = (1,1)
    # elif len(approx) == 5:
    #     cv2.putText(roi, "Pentagon", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0))
    # elif 6 < len(approx) < 15:
    #     cv2.putText(roi, "Ellipse", (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0))
    else:
        print('CIRCLE')
        triangle_direction = (1, 1)



    # corner detection
    # gray2 = np.float32(gray)
    # dst = cv2.cornerHarris(gray2, 2, 3, 0.04)
    #
    # dst = cv2.dilate(dst, None)
    #
    # img[dst > 0.01 * dst.max()] = [0, 0, 255]
    #
    # cv2.imshow('dst', img)
    #
    # cv2.imshow('hsv',hsv)
    cv2.imshow('bgr', bgr)
    cv2.imshow('marked', roi)
    cv2.waitKey(0)



img = cv2.imread('sha7.png')
img = cv2.resize(img,(250,250))
shape_detection(img,120,255,255)

img2 = cv2.imread('sha3.png')
img2 = cv2.resize(img2,(250,250))
shape_detection(img2,120,255,255)




cv2.destroyAllWindows()