import gym
import pix_sample_arena
import numpy as np
import os
import time
import math
import pybullet as p
import cv2
# import image_processing as impr
# import dict

###################

def color_recognise(h, s, v):
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
    if (h == 0 and s == 0 and v > 100):
        color_code = 0
    if (h == 0 and s == 255 and v > 100):
        color_code = 1
    if (h == 30 and s == 255 and v > 100):
        color_code = 2
    if (h == 60 and s == 255 and v > 100):
        color_code = 3
    if (h == 90 and s == 255 and v > 100):
        color_code = 4
    if (h == 120 and s == 255 and v > 100):
        color_code = 5
    if (h == 150 and s > 100 and v > 100):
        color_code = 6
    if (h == 60 and s == 255 and v == 91):
        color_code = 7

    return color_code


def shape_detection(roi, h, s, v):
    # roi = cv2.resize(roi, (250, 250))
    lower = np.array([h - 3, s - 3, v - 3])
    upper = np.array([h, s, v])
    '''
    :param roi: IMAGE OF SINGLE TILE
    :param h: HUE OF BLUE           120
    :param s: SATURATION OF BLUE    255
    :param v: VALUE OF BLUE         255

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
        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        # cv2.drawContours(roi, [approx], 0, (0), 1)
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        for i in range(3):
            triangle_corners[i][0] = approx.ravel()[2 * i]
            triangle_corners[i][1] = approx.ravel()[2 * i+1]

        # print(cv2.contourArea(cnt), end=' ')
        if (area[0] < cv2.contourArea(cnt) and cv2.contourArea(cnt) < w * h - 1000):
            area[0] = cv2.contourArea(cnt)
            area[1] = len(approx)
            # print(area[1])

    if area[1] == 3:
        print('waah bete waah')
        shape_code = 1
        # triangle direction detection
        # triangle_corners
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
    return shape_code, triangle_direction


def image_processing1(image, arena_dimensions):
    cv2.imshow('image_procssing1-read', image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # cv2.imshow('hsv',hsv)
    # fig, ax = plt.subplots(figsize=(10, 7))
    # plt.hist(image.flat,bins=100, range=(0,255))
    # plt.show()

    arena_information = np.zeros((arena_dimensions[0], arena_dimensions[1], 2), int)
    shapes_information = np.zeros((arena_dimensions[0], arena_dimensions[1], 2), int)
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

    tiles_centres = np.full((arena_dimensions[0], arena_dimensions[1], 2), -1, int)
    # print(tiles_centres)

    col = image.shape[0] // arena_dimensions[0]
    width = image.shape[1] // arena_dimensions[1]

    # for i in range(arena_dimensions[0]):
    #     image = cv2.line(image,(0,i*col),(image.shape[0],i*col),(0,255,0),2)
    #
    # for i in range(arena_dimensions[1]):
    #     image = cv2.line(image,(i*width,0),(i*width,image.shape[1]),(0,255,0),2)
    # cv2.imshow("marklines",image)

    l = np.array([0, 0, 90])
    u = np.array([180, 255, 255])
    mask = cv2.inRange(hsv, l, u)
    cv2.imshow("mask",mask)

    kernel = np.ones((5, 5), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    ###REDUNDANT MORPHOLOGICAL OPERATIONS
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area_mask = []
    for cnt in contours:
        area_mask.append(int(cv2.contourArea(cnt)))

    ###IN USE MORPHOLOGICAL OPERATIONS
    contours2, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area_opcl = []
    box_length = 0
    count = 0

    for cnt in contours2:
        area_opcl.append(int(cv2.contourArea(cnt)))

        box_length = math.floor(math.sqrt(area_opcl[0]))
        print(box_length)

        x, y, w, h = cv2.boundingRect(cnt)
        centre_x = x + w // 2
        centre_y = y + h // 2
        # cv2.putText(image,"FOUND",(x,centre_y),cv2.FONT_HERSHEY_COMPLEX,0.4,(0,0,0))

        ##################START ASSIGNING VALUES OF ARRAY
        print(hsv[y + 1, x + 1, 0], hsv[y + 1, x + 1, 1], hsv[y + 1, x + 1, 2])
        print(color_recognise(hsv[y + 1, x + 1, 0], hsv[y + 1, x + 1, 1], hsv[y + 1, x + 1, 2]))

        colorcode = str(color_recognise(hsv[y + 1, x + 1, 0], hsv[y + 1, x + 1, 1], hsv[y + 1, x + 1, 2]))
        # colorcode = 'STEVE'

        if (color_recognise(hsv[centre_y, centre_x, 0], hsv[centre_y, centre_x, 1], hsv[centre_y, centre_x, 2]) == 5):
            # cv2.putText(image, '5', (centre_x, centre_y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255))
            roi = image[y + 1:y + box_length, x + 1:x + box_length]
            # text= str(count)
            # cv2.imshow(text,roi)
            # cv2.waitKey(10000)

            # shapecode = str(count)
            # shapecode=str(shape_detection(roi, 120, 255, 227))#227 to 255

            # print(shape_detection(roi,120,255,255),count)
            count += 1
            # cv2.putText(image, shapecode, (centre_x, centre_y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,0))

        for i in range(arena_dimensions[0]):
            if (i * col < centre_y and (i + 1) * col > centre_y):
                y_coordinate = i
        for i in range(arena_dimensions[1]):
            if (i * width < centre_x and (i + 1) * width > centre_x):
                x_coordinate = i

        print(y_coordinate,x_coordinate)

        colorcode = str(y_coordinate) + str(x_coordinate)
        # cv2.putText(image,colorcode, (x, centre_y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255))

        ###upar pooree testing karlee fraaaanzzz

        arena_information[y_coordinate][x_coordinate][0] = color_recognise(hsv[y + 1, x + 1, 0], hsv[y + 1, x + 1, 1],
                                                                           hsv[y + 1, x + 1, 2])
        if (color_recognise(hsv[centre_y, centre_x, 0], hsv[centre_y, centre_x, 1], hsv[centre_y, centre_x, 2]) == 5):
            roi = image[y + 1:y + box_length, x + 1:x + box_length]
            arena_information[y_coordinate][x_coordinate][1], shapes_information[y_coordinate][
                x_coordinate] = shape_detection(roi, 120, 255, 227)  ###THIS VALUE NEEDS TO BE ADJUSTED
        tiles_centres[y_coordinate][x_coordinate] = (centre_y, centre_x)
        ##################
        # cv2.putText(image, colorcode, (x, centre_y), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255))

    mask = cv2.medianBlur(mask, 1)
    # cv2.imshow('BLACK REMOVED', mask)
    cv2.imshow('MARKED', image)
    cv2.imshow('closing', closing)

    length1 = len(area_mask)
    length2 = len(area_opcl)
    print(length1, length2, area_opcl[0])
    # for i in range(98):
    #     print(area_mask[i])

    # for i in range(length):
    #     dif = area_opcl[i]-area_mask[i]
    #     if dif != 0:
    #         print(i," : ",dif)
    cv2.waitKey(0)
    return arena_information, box_length, tiles_centres, shapes_information


####################


def graph_dict(arena_information,arena_dimensions,weight,shape_information):
    y_max=arena_dimensions[0]
    x_max=arena_dimensions[1]
    d={}
    for j in range(y_max):
        for i in range(x_max):

            if arena_information[j][i][0] in[0,1,2,3,4,5,6,7]:
                d['d' + str(j) + str(i)] = {}
                #LOWER TILE
                if j + 1 < y_max:
                    color_code=arena_information[j+1][i][0]
                    shape_code=arena_information[j+1][i][1]
                    shape_dir=shape_information[j+1][i]
                    if color_code in[0,1,2,3,4,5,6,7]:
                        if shape_code == 1:
                            # print("anti parallel10")
                            move_vector = np.array([1,0],int)
                            dot_product = move_vector[0]*shape_dir[0] + move_vector[1]*shape_dir[1]
                            if dot_product >= 0:
                                print("mauj kardi "+str(j)+str(i))
                                d['d' + str(j) + str(i)]['d' + str(j + 1) + str(i)] = weight[color_code]

                        else:
                            #yaha 2nd patient aur 2nd hospital ko nahi cross karne wali condition
                            # d['d' + str(j) + str(i)] = {}
                            d['d' + str(j) + str(i)]['d'+str(j+1)+str(i)]=weight[color_code]

                #UPPER TILE\
                if j-1 > -1:
                    color_code=arena_information[j-1][i][0]
                    shape_code=arena_information[j-1][i][1]
                    shape_dir = shape_information[j - 1][i]
                    if color_code in [0, 1, 2, 3, 4, 5, 6, 7]:
                        if shape_code == 1:
                            # print("anti parallel-10")
                            move_vector = np.array([-1, 0],int)
                            dot_product = move_vector[0]*shape_dir[0] + move_vector[1]*shape_dir[1]
                            if dot_product >= 0:
                                print("mauj kardi "+str(j)+str(i))
                                d['d' + str(j) + str(i)]['d' + str(j - 1) + str(i)] = weight[color_code]

                        else:
                            # yaha 2nd patient aur 2nd hospital ko nahi cross karne wali condition
                            # d['d' + str(j) + str(i)] = {}
                            d['d' + str(j) + str(i)]['d' + str(j - 1) + str(i)] = weight[color_code]

                #RIGHT TILE
                if i + 1 < x_max:
                    color_code=arena_information[j][i+1][0]
                    shape_code=arena_information[j][i+1][1]
                    shape_dir = shape_information[j][i + 1]
                    if color_code in [0, 1, 2, 3, 4, 5, 6, 7]:
                        if shape_code == 1:
                            # print("anti parallel01")
                            move_vector = np.array([0, 1],int)
                            dot_product = move_vector[0]*shape_dir[0] + move_vector[1]*shape_dir[1]
                            if (dot_product >= 0):
                                print("mauj kardi "+str(j)+str(i))
                                d['d' + str(j) + str(i)]['d' + str(j) + str(i+1)] = weight[color_code]

                        else:
                            # yaha 2nd patient aur 2nd hospital ko nahi cross karne wali condition
                            # d['d' + str(j) + str(i)] = {}
                            d['d' + str(j) + str(i)]['d' + str(j) + str(i+1)] = weight[color_code]

                #LEFT TILE
                if i - 1 > -1:
                    color_code=arena_information[j][i-1][0]
                    shape_code=arena_information[j][i-1][1]
                    shape_dir = shape_information[j][i - 1]
                    if color_code in [0, 1, 2, 3, 4, 5, 6, 7]:
                        if shape_code == 1:
                            # print("anti parallel0-1")
                            move_vector = np.array([0, -1],int)
                            dot_product = move_vector[0]*shape_dir[0] + move_vector[1]*shape_dir[1]
                            if dot_product >= 0:
                                print("mauj kardi "+str(j)+str(i))
                                d['d' + str(j) + str(i)]['d' + str(j) + str(i-1)] = weight[color_code]

                        else:
                            # yaha 2nd patient aur 2nd hospital ko nahi cross karne wali condition
                            # d['d' + str(j) + str(i)] = {}
                            d['d' + str(j) + str(i)]['d' + str(j) + str(i-1)] = weight[color_code]

                # d['d'+str(j)+str(i)]=d+str(j)+str(i)
    return d


def path_dijkstra(graph,start,goal):
    shortest_distance = {}
    track_predecessor = {}
    unseenNodes = graph
    infinity = 999999
    track_path = []

    for node in unseenNodes:
        shortest_distance[node] = infinity
    shortest_distance[start] = 0

    while unseenNodes:

        min_distance_node = None

        for node in unseenNodes:
            if min_distance_node is None:
                min_distance_node = node
            elif shortest_distance[node] < shortest_distance[min_distance_node]:
                min_distance_node = node

        path_options = graph[min_distance_node].items()

        for child_node, weight in path_options:

            if weight + shortest_distance[min_distance_node] < shortest_distance[child_node]:
                shortest_distance[child_node] = weight + shortest_distance[min_distance_node]
                track_predecessor[child_node] = min_distance_node

        unseenNodes.pop(min_distance_node)

    currentNode = goal

    while currentNode != start:
        try:
            track_path.insert(0,currentNode)
            currentNode = track_predecessor[currentNode]
        except KeyError:
            print("Path is not reachable")
            break
    if shortest_distance[goal] != infinity:
        print("Shortest distance is " + str(shortest_distance[goal]))
        print("Optimal Path is " + str(track_path))



##################################
###################################
##################################
parent_path = os.path.dirname(os.getcwd())
os.chdir(parent_path)
env = gym.make("pix_sample_arena-v0")
ar_dim = np.array([6,6])





weights = np.zeros(8,int)
'''
    COLOR = COLOR CODE, WEIGHT
    WHITE  = 0, 1
    RED    = 1, 4
    YELLOW = 2, 3
    GREEN  = 3, 2
    CYAN   = 4, 1?
    BLUE   = 5, no need but still say 1
    PINK   = 6, 1?
    DULL GREEN = 7, 1
'''
weights[0] = 1
weights[1] = 4
weights[2] = 3
weights[3] = 2
weights[4] = 1#asdf
weights[5] = 1#aise hee,blue ke liye wieghts is meaningless
weights[6] = 1
weights[7] = 1












































#####################################################
#####################################################

#####################################################
#####################################################

#####################################################
#####################################################

#####################################################
#####################################################

#####################################################
#####################################################
i=0
while True:
    i+=1
    p.stepSimulation()
    if i == 1:
        env.remove_car()
        img = env.camera_feed()
        x, y = 93, 93
        w = img.shape[0]
        h = img.shape[1]
        roi = img[x:w - x, y:h - y]
        arena_info, tile_length, tile_centres,shape_info = image_processing1(roi, ar_dim)
        cv2.destroyAllWindows()
        print(shape_info)
        dict_graph = graph_dict(arena_info, ar_dim, weights, shape_info)
        print(dict_graph)
        path_dijkstra(dict_graph, 'd21', 'd41')
        cv2.waitKey(1)
        env.respawn_car()

    # print(dict)
    env.move_husky(0.2, 0.2, 0.2, 0.2)
#     cv2.imshow("img", img)
#     cv2.waitKey(1)
cv2.destroyAllWindows()