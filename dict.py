import numpy as np


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




ar_dim = (6,6)
arena_info = np.array([[[ 6,  0],
       [ 3,  0],
       [ 1,  0],
       [-1,  0],
       [ 2,  0],
       [ 4,  3]],[[ 3,  0],
       [-1,  0],
       [ 3,  0],
       [-1,  0],
       [ 0,  0],
       [ 3,  0]], [[2, 0],
       [3, 0],
       [0, 0],
       [1, 0],
       [2, 0],
       [3, 0]], [[-1,  0],
       [ 0,  1],
       [ 3,  0],
       [ 0,  0],
       [-1,  0],
       [ 0,  0]], [[ 3,  0],
       [ 0,  0],
       [ 0,  0],
       [-1,  0],
       [-1,  0],
       [ 0,  0]], [[ 4,  2],
       [-1,  0],
       [ 1,  0],
       [ 3,  0],
       [ 3,  0],
       [ 7,  0]]])

shape_info = np.array([[[0, 0],
       [0, 0],
       [0, 0],
       [0, 0],
       [0, 0],
       [1, 1]], [[0, 0],
       [0, 0],
       [0, 0],
       [0, 0],
       [0, 0],
       [0, 0]], [[0, 0],
       [0, 0],
       [0, 0],
       [0, 0],
       [0, 0],
       [0, 0]], [[ 0,  0],
       [-1,  0],
       [ 0,  0],
       [ 0,  0],
       [ 0,  0],
       [ 0,  0]], [[0, 0],
       [0, 0],
       [0, 0],
       [0, 0],
       [0, 0],
       [0, 0]], [[1, 1],
       [0, 0],
       [0, 0],
       [0, 0],
       [0, 0],
       [0, 0]]])

dict = graph_dict(arena_info,ar_dim,weights,shape_info)
print(dict)
