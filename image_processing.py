import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

img1 = cv2.imread('stage0.png')
img1 = cv2.resize(img1, (800,800))
ar_dim = np.zeros(2,int)#(y,x)
ar_dim = (6,6)

def color_recognise(h,s,v):
    '''

    :param h: hue of color
    :param s: saturation
    :param v: value

    WHITE  = (0,0,255)
    RED    = (0,255,255)
    YELLOW = (30,255,255)
    GREEN  = (60,255,255)
    CYAN   = (90,255,255)
    BLUE   = (120,255,255)
    PINK   = (150,255,255)
    DULL GREEN = (76,255,104)

    :return:
    WHITE  = 0
    RED    = 1
    YELLOW = 2
    GREEN  = 3
    CYAN   = 4
    BLUE   = 5
    PINK   = 6
    DULL GREEN = 7
    '''

    color_code = -1
    ####CALIBRATED FROM STAGE1 GOOGLE DRIVE####
    # if (h == 0 and s == 0 and v == 255):
    #     color_code = 0
    # if (h == 0 and s == 255 and v == 255):
    #     color_code = 1
    # if (h == 30 and s == 255 and v == 255):
    #     color_code = 2
    # if (h == 60 and s == 255 and v == 255):
    #     color_code = 3
    # if (h == 90 and s == 255 and v == 255):
    #     color_code = 4
    # if (h == 120 and s == 255 and v == 255):
    #     color_code = 5
    # if (h == 150 and s == 255 and v == 255):
    #     color_code = 6
    # if (h == 76 and s == 255 and v == 104):
    #     color_code = 7

    ####CALIBRATED FROM SAMPLE ARENA####
    if (h == 0 and s == 0 and v >100):
        color_code = 0
    if (h == 0 and s == 255 and v >100):
        color_code = 1
    if (h == 30 and s == 255 and v >100):
        color_code = 2
    if (h == 60 and s == 255 and v >100):
        color_code = 3
    if (h == 90 and s == 255 and v >100):
        color_code = 4
    if (h == 120 and s == 255 and v >100):
        color_code = 5
    if (h == 150 and s >100 and v >100):
        color_code = 6
    if (h == 60 and s == 255 and v == 91):
        color_code = 7

    return color_code

def shape_detection(roi,h,s,v):
    roi = cv2.resize(roi, (250, 250))
    lower = np.array([h-3,s-3,v-3])
    upper = np.array([h,s,v])
    '''
        additionally it gives triangle direction
    :param roi: IMAGE OF SINGLE TILE
    :param h: HUE OF BLUE           120
    :param s: SATURATION OF BLUE    255
    :param v: VALUE OF BLUE         255/227

    :return:
    NOT FOUND = -1
    TRAINGLE    = 1
    SQUARE      = 2
    CIRCLE      = 3
    '''
    shape_code = -1
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(hsv, lower, upper)

    triangle_corners = np.zeros((3, 2), int)
    triangle_direction = np.zeros(2, int)  # (y,x)

    bgr_onlyblue = cv2.bitwise_and(bgr, bgr, mask=mask)
    gray_onlyblue = cv2.cvtColor(bgr_onlyblue, cv2.COLOR_BGR2GRAY)
    _, threshold_binary = cv2.threshold(gray_onlyblue, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = np.zeros(2)
    w = roi.shape[0]
    h = roi.shape[1]

    for cnt in contours:
        approx = cv2.approxPolyDP(cnt,  0.02* cv2.arcLength(cnt, True), True)
        cv2.drawContours(roi, [approx], 0, (0), 5)
        # cv2.imshow('shape_detect',roi)
        # cv2.waitKey(0)
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        for i in range(3):
            triangle_corners[i][0] = approx.ravel()[2 * i]
            triangle_corners[i][1] = approx.ravel()[2 * i+1]

        # print(cv2.contourArea(cnt), end=' ')
        if (area[0]<cv2.contourArea(cnt) and cv2.contourArea(cnt)<w*h-1000):
            area[0] = cv2.contourArea(cnt)
            area[1] = len(approx)
            # print(area[1])

    if area[1] == 3:
        shape_code = 1
        print('waah bete waah')
        #triangle direction detection
        #triangle_corners
        diff = np.zeros(6, int)
        diff[0] = abs(triangle_corners[2][0] - triangle_corners[1][0])  # 0
        diff[1] = abs(triangle_corners[0][0] - triangle_corners[2][0])  # 1
        diff[2] = abs(triangle_corners[0][0] - triangle_corners[1][0])  # 2

        diff[3] = abs(triangle_corners[2][1] - triangle_corners[1][1])  # 0
        diff[4] = abs(triangle_corners[0][1] - triangle_corners[2][1])  # 1
        diff[5] = abs(triangle_corners[0][1] - triangle_corners[1][1])  # 2

        indexofmin = np.where(diff == np.amin(diff))

        print(indexofmin[0][0], diff[indexofmin[0][0]])
        print(diff)

        if (indexofmin[0][0] <= 2 and indexofmin[0][0] >= 0):  # left,right
            triangle_direction[0] = 0  # (y,x)
            if indexofmin[0][0] == 0:
                triangle_direction[1] = np.sign(
                    triangle_corners[0][0] - (triangle_corners[2][0] + triangle_corners[1][0]) / 2)  # (x,y)

            if indexofmin[0][0] == 1:
                triangle_direction[1] = np.sign(
                    triangle_corners[1][0] - (triangle_corners[2][0] + triangle_corners[1][0]) / 2)  # (x,y)

            if indexofmin[0][0] == 2:
                triangle_direction[1] = np.sign(
                    triangle_corners[2][0] - (triangle_corners[0][0] + triangle_corners[1][0]) / 2)  # (x,y)

        if (indexofmin[0][0] <= 5 and indexofmin[0][0] >= 3):  # up,down
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



    elif area[1] == 4:
        shape_code = 2
        triangle_direction = (0, 0)

    elif area[1] > 4:
        shape_code = 3
        triangle_direction = (0, 0)
    return shape_code,triangle_direction



def image_processing1(image,arena_dimensions):
    cv2.imshow('image_procssing1-read',image)
    image2 = image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # cv2.imshow('hsv',hsv)
    # fig, ax = plt.subplots(figsize=(10, 7))
    # plt.hist(image.flat,bins=100, range=(0,255))
    # plt.show()

    arena_information = np.zeros((arena_dimensions[0],arena_dimensions[1],2),int)
    shapes_information = np.zeros((arena_dimensions[0],arena_dimensions[1],2),int)
    '''
    ARGUMENTS:
        Y COORDINATE (INT)
        X COORDINATE (INT)
        COLOR CODE   (INT)
        SHAPE CODE   (INT)
        
    BLACK  = -1
    WHITE  = 0
    RED    = 1
    YELLOW = 2
    GREEN  = 3
    CYAN   = 4
    BLUE   = 5
    PINK   = 6
    DULL GREEN = 7
    
    SHAPES
    NULL        = 0
    TRIANGLE    = 1
    SQUARE      = 2
    CIRCLE      = 3
    '''
    for j in range(arena_dimensions[0]):
        for i in range(arena_dimensions[1]):
            arena_information[j][i][0] = -1

    tiles_centres = np.full((arena_dimensions[0],arena_dimensions[1],2),-1,int)
    # print(tiles_centres)

    col = image.shape[0] // arena_dimensions[0]
    width = image.shape[1] // arena_dimensions[1]

    for i in range(arena_dimensions[0]):
        image = cv2.line(image,(0,i*col),(image.shape[0],i*col),(0,255,0),2)

    for i in range(arena_dimensions[1]):
        image = cv2.line(image,(i*width,0),(i*width,image.shape[1]),(0,255,0),2)



    l = np.array([0,0,90])
    u = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, l, u)

    kernel = np.ones((5,5), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    ###REDUNDANT MORPHOLOGICAL OPERATIONS
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area_mask=[]
    for cnt in contours:
        area_mask.append(int(cv2.contourArea(cnt)))


    ###IN USE MORPHOLOGICAL OPERATIONS
    contours2, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area_opcl = []
    box_length = 0
    count=0




    for cnt in contours2:
        area_opcl.append(int(cv2.contourArea(cnt)))

        box_length =math.floor( math.sqrt(area_opcl[0]) )
        print('length: ',box_length)

        x,y,w,h = cv2.boundingRect(cnt)
        centre_x= x+w//2
        centre_y= y+h//2
        # cv2.putText(image,"FOUND",(x,centre_y),cv2.FONT_HERSHEY_COMPLEX,0.4,(0,0,0))

        ##################START ASSIGNING VALUES OF ARRAY
        print(hsv[y+1,x+1,0],hsv[y+1,x+1,1],hsv[y+1,x+1,2])
        print(color_recognise(hsv[y+1,x+1,0],hsv[y+1,x+1,1],hsv[y+1,x+1,2]))

        colorcode = str(color_recognise(hsv[y+1,x+1,0],hsv[y+1,x+1,1],hsv[y+1,x+1,2]))
        cv2.putText(image2, colorcode, (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255))
        # colorcode = 'STEVE'

        if(color_recognise(hsv[centre_y, centre_x, 0], hsv[centre_y, centre_x, 1], hsv[centre_y, centre_x, 2])==5):
            # cv2.putText(image, '5', (centre_x, centre_y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255))
            roi = image[y+1:y+box_length, x+1:x+box_length]
            # text= str(count)
            # cv2.imshow(text,roi)
            # cv2.waitKey(10000)


            # shapecode = str(count)
            # shapecode=str(shape_detection(roi, 120, 255, 227))#227 to 255



            # print(shape_detection(roi,120,255,255),count)
            count += 1
            # cv2.putText(image, shapecode, (centre_x, centre_y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,0))

        for i in range(arena_dimensions[0]):
            if(i*col<centre_y and (i+1)*col>centre_y):
                y_coordinate = i
        for i in range(arena_dimensions[1]):
            if(i*width<centre_x and (i+1)*width>centre_x):
                x_coordinate = i

        colorcode = str(y_coordinate)+str(x_coordinate)
        # cv2.putText(image2,colorcode, (x, centre_y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255))



        ###upar pooree testing karlee fraaaanzzz

        arena_information[y_coordinate][x_coordinate][0]=color_recognise(hsv[y+1,x+1,0],hsv[y+1,x+1,1],hsv[y+1,x+1,2])
        #   shape detection and shape information
        if (color_recognise(hsv[centre_y, centre_x, 0], hsv[centre_y, centre_x, 1], hsv[centre_y, centre_x, 2]) == 5):
            cv2.imshow('hsv2roi',image2)
            roi = image[y + 1:y + box_length, x + 1:x + box_length]

            arena_information[y_coordinate][x_coordinate][1], shapes_information[y_coordinate][
                x_coordinate] = shape_detection(roi, 120, 255, 227)  ###THIS VALUE NEEDS TO BE ADJUSTED
            # shape_code = str(shapes_information[y_coordinate][x_coordinate])
            shape_code = str(arena_information[y_coordinate][x_coordinate][1])
            cv2.putText(image2, shape_code, (centre_x, centre_y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255))
        tiles_centres[y_coordinate][x_coordinate]=(centre_y,centre_x)

        ##################
        # cv2.putText(image, colorcode, (x, centre_y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255))


    mask = cv2.medianBlur(mask,1)
    # cv2.imshow('BLACK REMOVED', mask)
    cv2.imshow('MARKED', image2)
    # cv2.imshow('closing', closing)

    length1 = len(area_mask)
    length2 = len(area_opcl)
    print(length1,length2,area_opcl[0])
    # for i in range(98):
    #     print(area_mask[i])



    # for i in range(length):
    #     dif = area_opcl[i]-area_mask[i]
    #     if dif != 0:
    #         print(i," : ",dif)
    cv2.waitKey(0)
    return arena_information, box_length, tiles_centres, shapes_information



ar,_,ar2,ar3=image_processing1(img1,ar_dim)

b = list(ar)
print("dhappa")
print(ar3)


#pink color masks position in pixels



cv2.destroyAllWindows()