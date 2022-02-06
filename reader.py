from tkinter import *
from tkinter import messagebox as mb
from math import *
import numpy as np
from PIL import Image
img_folder = "img\\"
data_folder = "data\\"
def draw_pix(x, y):
    holst.create_line(x, y, x, y + 1, fill = 'black')
def draw_line(x_from, y_from, x_to, y_to):
    holst.create_line(x_from, y_from, x_to, y_to, fill = 'black')
def is_zero(matr):
    ans = 0
    for i in range(len(matr) - len(matr) % 3):
        ans += matr[i]
    
    return ans
def write_brain(image, out_file) :
    print("creating", out_file)
    image = Image.open(image)
    array = np.array(image)
    h_wid, h_height = image.size
    brain_left = []
    brain_right = []
    for i in range(h_wid):
        if i % 50 == 0:
            print(i)
        flag = 0
        for j in range (h_height - 50):
            if flag == 0:
                if (is_zero(array[j][i]) != 0):
                    #draw_pix(j, i)
                    if (j < i):
                        brain_left.append([i, j + 75])
                        flag = 1
                        ans_img = 0
            if flag == 1:
                for k in range(50):
                    ans_img += is_zero(array[j + k][i])
                if (ans_img == 0):
                    #draw_pix(j, i)
                    brain_right.append([i, j + 75 ])
                    break
                    #flag = 0
                ans = ans_img = 0
    for i in range(h_height - 50):
        if i % 50 == 0:
            print(i)
        flag = 0
        for j in range (h_wid):
            if flag == 0:
                if (is_zero(array[i][j]) != 0):
                    #draw_pix(j, i)
                    if (j < i):
                        if (i  < (h_height-50) //2):
                            if j not in brain_left:
                                brain_left.append([j , i + 75])
                        else:
                            if j not in brain_right:
                                brain_right.append([j , i + 75])
                        flag = 1
                        ans_img = 0
            if flag == 1:
                for k in range(j, h_wid - j, 1):
                    ans_img += is_zero(array[i][k])
                if (ans_img == 0):
                    if (i  < (h_height-50) //2):
                         if j not in brain_left:
                                brain_left.append([j , i + 75])
                    else:
                         if j not in brain_right:
                                brain_right.append([j , i + 75])
                    break
                    #flag = 0
                ans = ans_img = 0
    brain_left.sort(key = lambda x:x[0])
    brain_right = sorted(brain_right, key = lambda x: x[0], reverse = True)
    brain = brain_left + brain_right
    f = open(out_file, 'w')
    for i in range(len(brain)):
        f.write(str(brain[i][0]) + ' ' +  str(brain[i][1]) + '\n')
    f.close()
images = ["top.png","front.png", "front_plus_1.png", "front_plus_2.png", "front_plus_3.png", "front_plus_4.png", "front_plus_5.png"
          , "front_minus_1.png", "front_minus_2.png", "front_minus_3.png", "front_minus_4.png", "front_minus_5.png", "side.png"]
out_files = ["top.txt", "front.txt", "front_plus_1.txt", "front_plus_2.txt", "front_plus_3.txt", "front_plus_4.txt", "front_plus_5.txt"
             , "front_minus_1.txt", "front_minus_2.txt", "front_minus_3.txt", "front_minus_4.txt", "front_minus_5.txt", "side.txt"]
for i in range(len(images)):
    write_brain(img_folder + images[i], data_folder + out_files[i])
    

