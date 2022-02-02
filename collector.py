from tkinter import *
from tkinter import messagebox as mb
from math import *
import numpy as np
from PIL import Image, ImageDraw    #  pip install Pillow
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy
import random
data_folder = "data\\"
def make_parts_sovokup(parts_2d, other_parts2):
    sovokup_parts = {'horyzontal' : parts_2d, 'vertical' : other_parts2}
    return sovokup_parts
    
def extract_points(sovokup_parts, percentage):
    neurs = [[],[],[]]
    X = []
    Y = []
    Z = []
    Dots = []
    for i in range(len(sovokup_parts['horyzontal'])):
        rand_dots = random.sample(sovokup_parts['horyzontal'][i], len(sovokup_parts['horyzontal'][i]) // 100 * (percentage));
        rand_x = random.choice(sovokup_parts['horyzontal'][i])[0]
        neur = generate_neurone_dot_x(sovokup_parts['horyzontal'][i], rand_x)
        if (len(neur) > 0):
            neurs[0].append(neur[0]); neurs[1].append(neur[1]); neurs[2].append(neur[2]);
        for j in range(len(rand_dots)):
                #continue;
                #if (X.count(rand_dots[j][0]) == 0 and Y.count(rand_dots[j][1]) == 0 and Z.count(rand_dots[j][2]) == 0):
                    X.append(rand_dots[j][0])
                    Y.append(rand_dots[j][1])
                    Z.append(rand_dots[j][2])
                    Dots.append([rand_dots[j][0], rand_dots[j][1], rand_dots[j][2]])


                    
    for i in range(len(sovokup_parts['vertical'])):
        rand_dots = random.sample(sovokup_parts['vertical'][i], len(sovokup_parts['vertical'][i]) // 100 * (percentage))
        rand_y = random.choice(sovokup_parts['vertical'][i])[1]
        neur = generate_neurone_dot_y(sovokup_parts['vertical'][i], rand_y)
        if (len(neur) > 0):
            neurs[0].append(neur[0]); neurs[1].append(neur[1]); neurs[2].append(neur[2]);
        for j in range(len(rand_dots)):
            #if (X.count(rand_dots[j][0]) == 0 and Y.count(rand_dots[j][1]) == 0 and Z.count(rand_dots[j][2]) == 0):
                    X.append(rand_dots[j][0])
                    Y.append(rand_dots[j][1])
                    Z.append(rand_dots[j][2])
                    Dots.append([rand_dots[j][0], rand_dots[j][1], rand_dots[j][2]])
    Dots_copy = []
    for i in range(len(Dots)):
        if Dots[i] not in Dots_copy:
            Dots_copy.append(Dots[i])
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    return X, Y, Z, neurs, Dots_copy
def make_2d_parts(parts):
    parts_copy = []
    for i in range (len (parts)):
        for j in range (len(parts[i][0])):
            parts_copy.append([parts[i][0][j], parts[i][1][j], parts[i][2][j]])    
    parts_copy.sort(key = lambda x: x[1])
    new_parts = []
    y_ = parts_copy[0][1]
    one_part = []
    for i in range(len(parts_copy)):
        if (y_ != parts_copy[i][1]):
                new_parts.append(one_part)
                y_ = parts_copy[i][1]
                one_part = []
        one_part.append(parts_copy[i])
    return new_parts
def transform_arrays_into_dots(X, Y, Z):
    dots = []
    for i in range(len(X)):
        dots.append([X[i], Y[i], Z[i]])
    return dots
def len_between_dots(dot1, dot2):
    ans = sqrt ((dot2[0] - dot1[0])**2 + (dot2[1] - dot1[1])**2 + (dot2[2] - dot1[2])**2)
    return ans
def create_links_for_surface(dots):
    dots.sort(key = lambda x: x[2])
    links_surface = [[] * 1 for i in range(len(dots))]
    connections = 4
    links_matr = [[0] * len(dots) for i in range(len(dots))]
    for i in range(len(dots)):
        for k in range (connections):
            min_len = 10000000
            min_i = -1
            min_j = -1
            for j in range(len(dots)):
                if (i != j and links_matr[i][j] == 0):
                    len_ = len_between_dots(dots[i], dots[j])
                    if len_ < min_len:
                        min_len = len_
                        min_i = i
                        min_j = j
            if (min_i > -1 and min_j > -1 and min_len < 10000000):
                links_matr[i][min_j] = links_matr[min_j][i] = 1
                links_surface[i].append(dots[min_j])
                links_surface[min_j].append(dots[i])
    return links_surface
            
def draw_pix(x, y, holst):
    holst.create_line(x, y, x, y + 1, fill = 'black')
def extend_array(arr, len_):
    dif = len_ - len(arr)
    j = len(arr) - 1
    arr.extend([0] * dif)
    k = len(arr) - 1
    step = -2
    for i in range(dif):
        arr[k] = arr[j]
        arr[k - 1] = arr[j]
        j -= 1
        k += step
def draw_line(x_from, y_from, x_to, y_to, holst):
    holst.create_line(x_from, y_from, x_to, y_to, fill = 'black')
def is_zero(matr):
    ans = 0
    for i in range(len(matr)):
        ans += matr[i]
    return ans
def get_simmetry_x(x, xc):
    if (x < xc):
        mnoj = 1
    else:
        mnoj = -1
    dif = fabs(x - xc)
    ans = x + mnoj * (dif * 2)
    return ans
def get_0(elem):
    return elem[0]
def get_1(elem):
    return elem[1]
def get_2(elem):
    return elem[2]
def get_len(elem):
    return len(elem[0])
def get_wid(view):
    max_w = max(view, key = get_0)[0]
    min_w = min(view, key = get_0)[0]
    return int(max_w - min_w)
def get_height(view):
    max_h = max(view, key = get_1)[1]
    min_h = min(view, key = get_1)[1]
    return int(max_h - min_h)

def get_height_XZ(view):
    max_h = max(view, key = get_2)[2]
    min_h = min(view, key = get_2)[2]
    return int(max_h - min_h)
def get_x(listt):
    x = []
    for i in listt:
        x.append(i[0])
    return x
def get_y(listt):
    x = []
    for i in listt:
        x.append(i[1])
    return x
def get_np(array, leng):
    
    x = []
    for i in range (leng):
        x.append(array[i % len(array)])
    x = np.array(x)
    return x
def find_y_stop(y_start, lim_view, main_view):
    wid_lim = get_wid(lim_view)
    for i in range(len(main_view)-1):
        for j in range(i + 1, len(main_view)):
            if (main_view[i][1] == main_view[j][1] and main_view[i][1] > y_start):
                #print(main_view[i][0], lim_val)
                if (max(main_view[i][0], main_view[j][0]) - min(main_view[i][0], main_view[j][0]) - get_wid(lim_view) < 5):
                    return main_view[i][1];
def get_center(dots):
    x_left = min(dots, key = get_0)[0]
    x_right = max(dots, key = get_0)[0]
    
    y_left = min(dots, key = get_1)[1]
    y_right = max(dots, key = get_1)[1]
    xc = (x_right + x_left) // 2
    yc = (y_right + y_left) // 2
    return xc, yc
def get_center_XZ(dots):
    x_left = min(dots, key = get_0)[0]
    x_right = max(dots, key = get_0)[0]
    
    z_left = min(dots, key = get_2)[2]
    z_right = max(dots, key = get_2)[2]
    xc = (x_right + x_left) // 2
    zc = (z_right + z_left) // 2
    return xc, zc
def scale_fig(dots, kx, ky):
        xc, yc = get_center(dots)
        dots_copy = copy.deepcopy(dots)
        for i in range(len(dots_copy)):
                dots_copy[i][0] = int(float(dots_copy[i][0]) * kx + (1 - kx) * float(xc))
                dots_copy[i][1] = int(float(dots_copy[i][1]) * ky + (1 - ky) * float(yc))
        return dots_copy
def scale_fig_XZ(dots, kx, kz):
        xc, zc = get_center_XZ(dots)
        dots_copy = copy.deepcopy(dots)
        for i in range(len(dots_copy)):
                dots_copy[i][0] = int(float(dots_copy[i][0]) * kx + (1 - kx) * float(xc))
                dots_copy[i][2] = int(float(dots_copy[i][2]) * kz + (1 - kz) * float(zc))
        return dots_copy
def correction_sizes(brain):
    min_x = min(get_wid(brain['top']), get_wid(brain['front']))
    brain['top'] = scale_fig(brain['top'], min_x / get_wid(brain['top']), 1)
    brain['front'] = scale_fig(brain['front'], min_x / get_wid(brain['front']), 1)

    min_y = min(get_wid(brain['side']), get_height(brain['top']))
    brain['top'] = scale_fig(brain['top'], 1, min_y / get_height(brain['top']))
    brain['side'] = scale_fig(brain['side'], min_y / get_wid(brain['side']), 1)

    min_z = min(get_height(brain['side']), get_height(brain['front']))
    brain['front'] = scale_fig(brain['front'], 1, min_z / get_height(brain['front']))
    brain['side'] = scale_fig(brain['side'], 1, min_z / get_height(brain['side']))
def get_coeff_front(copy_top, main_wid, i):
          #print("search for ", i)
          for j in range(len(copy_top) - 1):
                    if (int(copy_top[j][1]) == i):
                        for k in range(j + 1, len(copy_top)):
                            #print(copy_top[j][1], copy_top[k][1])
                            if (copy_top[k][1] == i) :
                                wid = get_diff(copy_top[k][0], copy_top[j][0])
                                #print(wid, main_wid)
                                return float(wid / main_wid)
          return -1;
def find_side_part(view, x):
    for i in range(len(view)):
        if (view[i][0] == x):
            y1 = view[i][1]
            for i in range(i, len(view)):
                if (view[i][0] == x):
                    y2 = view[i][1]
                    break
            break
    return (max(y1,y2) - min(y1,y2))
                
def get_diff(dot_1, dot_2):
    return max(dot_1, dot_2) - min(dot_1, dot_2)
# DODEEEELAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAATTTTTTTTTTTTTT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def correcting_height(parts):

    while i <  (len(parts) - 1):
        print(i, len(parts) - 1)
        distance = abs(parts[i][1][0] -  parts[i + 1][1][0])
        if (distance > step and parts[i][1][0] <= parts[i + 1][1][0]):
            print(parts[i][1][0] , parts[i + 1][1][0]," are to be scaled")
            cur_y = parts[i][1][0]
            wid_1 = get_wid(parts[i + 1])
            wid_2 = get_wid(parts[i])

            
            
            amount = distance // step
            mnoj = 1
            cur_index = i
            

            if (wid_1 < wid_2):
                koeff = wid_1 / wid_2
                step_koeff = fabs(1 - koeff) / amount
                
                koeff_cyc = 1
                for j in range (amount):
                    cur_index +=  1
                    cur_y = cur_y + step
                    koeff_cyc -= step_koeff
                    one_part = copy.deepcopy(parts[i])
                    #one_part = scale_fig(one_part, koeff_cyc, koeff_cyc)
                    one_part = scale_fig_XZ(one_part, koeff_cyc, koeff_cyc)
                    for k in range (len(one_part[1])):
                        one_part[1][k] = cur_y
                    #parts.insert(cur_index, one_part)
                    parts_added.append(one_part)
            else:
                koeff = wid_1 / wid_2
                step_koeff = fabs(1 - koeff) / amount
                koeff_cyc = 1
                for j in range (amount):
                    cur_index +=  1
                    cur_y = cur_y + step
                    koeff_cyc += step_koeff
                    
                    one_part = copy.deepcopy(parts[i])
                    
                    #one_part = scale_fig(one_part, koeff_cyc, koeff_cyc)
                    one_part = scale_fig_XZ(one_part, koeff_cyc, koeff_cyc)
                    for k in range (len(one_part[1])):
                        one_part[1][k] = cur_y
                    #parts.insert(cur_index, one_part)
                    parts_added.append(one_part)
            #i = cur_index
        #else:
        i += 1

    print("ADDED PART IS")
    for i in parts_added:
        print(i[1][0])
    parts += parts_added
    parts.sort(key = get_1)

def closing_distance(parts):

    i = 0
    step = 6
    parts_added = []
    while i <  (len(parts) - 1):
        print(i, len(parts) - 1)
        distance = abs(parts[i][1][0] -  parts[i + 1][1][0])
        if (distance > step and parts[i][1][0] <= parts[i + 1][1][0]):
            print(parts[i][1][0] , parts[i + 1][1][0]," are to be scaled")
            cur_y = parts[i][1][0]
            wid_1 = get_wid(parts[i + 1])
            wid_2 = get_wid(parts[i])

            
            
            amount = distance // step
            mnoj = 1
            cur_index = i
            

            if (wid_1 < wid_2):
                koeff = wid_1 / wid_2
                step_koeff = fabs(1 - koeff) / amount
                
                koeff_cyc = 1
                for j in range (amount):
                    cur_index +=  1
                    cur_y = cur_y + step
                    koeff_cyc -= step_koeff
                    one_part = copy.deepcopy(parts[i])
                    #one_part = scale_fig(one_part, koeff_cyc, koeff_cyc)
                    one_part = scale_fig_XZ(one_part, koeff_cyc, koeff_cyc)
                    for k in range (len(one_part[1])):
                        one_part[1][k] = cur_y
                    #parts.insert(cur_index, one_part)
                    parts_added.append(one_part)
            else:
                koeff = wid_1 / wid_2
                step_koeff = fabs(1 - koeff) / amount
                koeff_cyc = 1
                for j in range (amount):
                    cur_index +=  1
                    cur_y = cur_y + step
                    koeff_cyc += step_koeff
                    
                    one_part = copy.deepcopy(parts[i])
                    
                    #one_part = scale_fig(one_part, koeff_cyc, koeff_cyc)
                    one_part = scale_fig_XZ(one_part, koeff_cyc, koeff_cyc)
                    for k in range (len(one_part[1])):
                        one_part[1][k] = cur_y
                    #parts.insert(cur_index, one_part)
                    parts_added.append(one_part)
            #i = cur_index
        #else:
        i += 1

    print("ADDED PART IS")
    for i in parts_added:
        print(i[1][0])
    parts += parts_added
    parts.sort(key = get_1)

def plot_front(brain,ax_3d, fig):
    xc,yc = get_center(brain['top'])
    print(xc, yc, " - CENTER")
    copy_top = copy.deepcopy(brain['top'])
    copy_top.sort(key = get_1)
    ind = 20; step = 5; y_next = copy_top[0][1] + step; minuses_ind = 0; pluses_ind = 0
    minuses = [copy.deepcopy(brain['front_minus_5']), copy.deepcopy(brain['front_minus_4']), copy.deepcopy(brain['front_minus_3']), copy.deepcopy(brain['front_minus_2']), copy.deepcopy(brain['front_minus_1']), copy.deepcopy(brain['front'])]
    pluses = [copy.deepcopy(brain['front_plus_1']), copy.deepcopy(brain['front_plus_2']), copy.deepcopy(brain['front_plus_3']), copy.deepcopy(brain['front_plus_4']), copy.deepcopy(brain['front_plus_5'])]
    cur_len_slice = get_wid(minuses[0])
    wid_top = get_wid(brain['top'])
    j = 0;
    while j < len(copy_top):
        cur_y = copy_top[j][1]
        cur_line = []
        while(j < len(copy_top) and copy_top[j][1] == cur_y):
                cur_line.append(copy_top[j])
                j += 1
                if (len(cur_line) <= 1):
                    if (len(cur_line) < 1): 
                        continue
                    else:
                        appender_x = get_simmetry_x(cur_line[0][0], xc) 
                        cur_line.append([appender_x, cur_line[0][1]])
                if (get_wid(cur_line) >= wid_top):
                    wid_top_index = j - len(cur_line)
                    break
    print(copy_top[wid_top_index], " SHIROTA")
    front_parts = []
    one_part = []
    one_part.append([xc])
    one_part.append([copy_top[0][1]])
    one_part.append([0])
    front_parts.append(one_part)
    while (ind <= wid_top_index):
        if (abs(copy_top[ind][1] - y_next) < 2):
            cur_line = []
            while(ind < len (copy_top) - 1  and (abs(copy_top[ind][1] - y_next) < 2)):
                cur_line.append(copy_top[ind])
                ind += 1
            ind -= 1
            if (len(cur_line) <= 1):
                if (len(cur_line) < 1): 
                    continue
                else:
                    appender_x = get_simmetry_x(cur_line[0][0], xc) 
                    cur_line.append([appender_x, cur_line[0][1]])
            y_next += step
            cur_len = get_wid(cur_line)
            if (cur_len == 1):
                continue
            if (cur_len <= cur_len_slice):
                koeff = cur_len / cur_len_slice

                front_part = scale_fig(minuses[minuses_ind], koeff, koeff)

                one_part = []
                one_part.append(get_x(front_part))
                one_part.append([cur_line[0][1]] * len(front_part))
                one_part.append(get_y(front_part))
                while (one_part[2][0] < one_part[2][len(one_part[2]) - 1]):
                    one_part[0].append(one_part[0].pop(0))
                    one_part[1].append(one_part[1].pop(0))
                    one_part[2].append(one_part[2].pop(0))
                

                x_front = get_np(one_part[0],len(one_part[0]))
                y_front = get_np(one_part[1],len(one_part[1]))
                z_front = get_np(one_part[2],len(one_part[2]))
    
                front_parts.append(one_part)
                #ax_3d.plot(x_front, y_front, z_front, color = 'm')
            else:
                if (minuses_ind + 1 < len(minuses)):
                    minuses_ind += 1
                cur_len_slice = get_wid(minuses[minuses_ind])
        elif ((copy_top[ind][1] > y_next)):
            while (copy_top[ind][1] > y_next):
                y_next += step
        ind += 1
    cur_len_slice = get_wid(pluses[0]); y_max = max(copy_top, key = get_1)[1];
    for i in range(len(copy_top)):
        if copy_top[i][1] == y_max:
            max_ind = i
            break
    while (ind < max_ind):
        if (abs(copy_top[ind][1] - y_next) < 2):
            cur_line = []
            while(ind < max_ind - 1  and (abs(copy_top[ind][1] - y_next) < 2)):
                cur_line.append(copy_top[ind])
                ind += 1
            ind -= 1
            if (len(cur_line) <= 1):
                if (len(cur_line) < 1):
                    ind += 1
                    continue
                else:
                    appender_x = get_simmetry_x(cur_line[0][0], xc) 
                    cur_line.append([appender_x, cur_line[0][1]])
            #print(cur_line)
            y_next += step
            cur_len = get_wid(cur_line)
            if (cur_len == 1):
                continue
            if (cur_len > cur_len_slice):
                koeff = cur_len / cur_len_slice

             #   print("koeff = ", koeff)
              #  print("cur_len = ",cur_len, "cur slice len= ", cur_len_slice)
                front_part = scale_fig(pluses[pluses_ind], koeff, koeff)

                one_part = []
                one_part.append(get_x(front_part))
                one_part.append([cur_line[0][1]] * len(front_part))
                one_part.append(get_y(front_part))
                while (one_part[2][0] < one_part[2][len(one_part[2]) - 1]):
                    one_part[0].append(one_part[0].pop(0))
                    one_part[1].append(one_part[1].pop(0))
                    one_part[2].append(one_part[2].pop(0))
                

                x_front = get_np(one_part[0],len(one_part[0]))
                y_front = get_np(one_part[1],len(one_part[1]))
                z_front = get_np(one_part[2],len(one_part[2]))

                    
                front_parts.append(one_part)
                #ax_3d.plot(x_front, y_front, z_front, color = 'm')
            else:
                if (pluses_ind + 1 < len(pluses)):
                    pluses_ind += 1
                    cur_len_slice = get_wid(pluses[pluses_ind])
                elif (pluses_ind == len(pluses) - 1):
                    
                    front_part = scale_fig(pluses[pluses_ind], koeff, koeff)

                    one_part = []
                    one_part.append(get_x(front_part))
                    one_part.append([cur_line[0][1]] * len(front_part))
                    one_part.append(get_y(front_part))
                    while (one_part[2][0] < one_part[2][len(one_part[2]) - 1]):
                        one_part[0].append(one_part[0].pop(0))
                        one_part[1].append(one_part[1].pop(0))
                        one_part[2].append(one_part[2].pop(0))
                    

                    x_front = get_np(one_part[0],len(one_part[0]))
                    y_front = get_np(one_part[1],len(one_part[1]))
                    z_front = get_np(one_part[2],len(one_part[2]))

                    front_parts.append(one_part)

                    #ax_3d.plot(x_front, y_front, z_front, color = 'm') 
                    koeff = cur_len / cur_len_slice
                    #print("KOEFF = ", koeff)     
                else:
                    cur_len_slice = get_wid(pluses[pluses_ind])
        elif ((copy_top[ind][1] > y_next)):
            while (copy_top[ind][1] > y_next):
                y_next += step
        ind += 1
    one_part = []
    one_part.append([xc])
    one_part.append([copy_top[max_ind][1]])
    one_part.append([0])
    front_parts.append(one_part)
    return front_parts
    
def plot_side(brain, ax_3d, fig):
    xc,yc = get_center(brain['top'])
    print(xc,yc, "are top centers")
    x_start = min(brain['top'], key =get_0)[0] + 1; x_stop = max(brain['top'],key= get_0)[0] - 1; step = 10; wid = get_height(brain['top']);
    flag_start = 0; flag_stop = 0;
    for i in range(len(brain['top'])):
        if (brain['top'][i][0] == x_start):
            if (flag_start == 0):
                i_start1 = i
                flag_start = 1
            else:
                i_start2 = i
        if (brain['top'][i][0] == x_stop):
            if (flag_stop == 0):
                i_stop1 = i
                flag_stop = 1
            else:
                i_stop2 = i
    i_start1, i_stop1 = min(i_start1, i_stop1), max(i_start1, i_stop1)
    i_start2, i_stop2 = min(i_start2, i_stop2), max(i_start2, i_stop2)
    side_parts = []
    while(i_start1 < i_stop1) and (i_start2 < i_stop2):
        koef = (max(brain['top'][i_start1][1], brain['top'][i_start2][1]) - min(brain['top'][i_start1][1], brain['top'][i_start2][1])) / wid
        side_part = scale_fig (brain['side'], koef, koef)
        side_parts.append(side_part)
        x_side = get_np(([brain['top'][i_start1][0]] * len(side_part)),len(side_part))
        y_side = get_np(get_x(side_part),len(side_part))
        z_side = get_np(get_y(side_part),len(side_part))
        #ax_3d.plot(x_side, y_side, z_side, color = 'c')
                
        i_start1 += step
        i_start2 += step
    return side_parts
def plot_top(brain, ax_3d, fig):
    

    x_top = get_np(get_x(brain['top']),len(brain['top']))
    y_top = get_np(get_y(brain['top']),len(brain['top']))
    z_top = get_np(([0] * len(brain['top'])),len(brain['top']))

    ax_3d.plot(x_top, y_top, z_top, color = 'c')
def get_plots(brain):
    fig = plt.figure(figsize=(7, 4))
    
    ax_3d = fig.add_subplot(projection='3d')

    plot_top(brain, ax_3d, fig)

    #side_parts = plot_side(brain, ax_3d, fig)

    front_parts = plot_front(brain, ax_3d, fig)

    #fig.show()
    return front_parts
def get_min(parts, ind):
    if (ind == 0):
        f = get_0
    elif (ind == 1):
        f = get_1
    elif (ind == 2):
        f = get_2
    min_ = parts[0][ind][0]
    for i in parts:
        min_i = min(i[ind])
        if (min_i < min_):
            min_ = min_i
    return min_
def get_max(parts, ind):
    if (ind == 0):
        f = get_0
    elif (ind == 1):
        f = get_1
    elif (ind == 2):
        f = get_2
    max_ = max(parts[0][ind])
    for i in parts:
        max_i = max(i[ind])
        if (max_i > max_):
            max_ = max_i
    return max_
#NEURONS
def get_neur_links(neurs):
    links = []
    for i in range(len(neurs[0]) - 1):
        dot1 = [neurs[0][i], neurs[1][i], neurs[2][i]]
        for j in range(i + 1, len(neurs[0])):
            dot2 = [neurs[0][j], neurs[1][j], neurs[2][j]]
            links.append([dot1, dot2])
    return links
def generate_neurone_dot_x(slice_, x):
    buf = []
    for i in slice_:
        if i[0] == x:
            buf.append(i)
    #print("BUF = ", buf)
    min_y = min(buf, key = lambda x: x[1])[1]
    max_y = max(buf, key = lambda x: x[1])[1]

    min_z = min(buf, key = lambda x: x[2])[2]
    max_z = max(buf, key = lambda x: x[2])[2]
    neur = []
    if (len(buf) >= 2):
        neur = [x, random.randint(min_y, max_y), random.randint(min_z, max_z)]
    return neur
def generate_neurone_dot_y(slice_, y):
    buf = []
    for i in slice_:
        if i[1] == y:
            buf.append(i)
   # print("BUF = ", buf)
    min_x = min(buf, key = lambda x: x[0])[0]
    max_x = max(buf, key = lambda x: x[0])[0]

    min_z = min(buf, key = lambda x: x[2])[2]
    max_z = max(buf, key = lambda x: x[2])[2]
    neur = []
    if (len(buf) >= 2):
        neur = [random.randint(min_x, max_x), y, random.randint(min_z, max_z)]
    return neur
# В следующих  функциях работаем с новой СД для brain, по вертикали соединяем  ребра по процентам от длины
def new_parts_sort2(one_part):
    one_part.sort(key = get_1)
    one_part.reverse()
    z_buf = copy.deepcopy(one_part)
    z_buf.sort(key = lambda x: x[2])
    down = []
    up = []
    flag = 0
    for i in range(len(one_part)):
        for j in range(len(z_buf)):
            if (z_buf[j][1] == one_part[i][1]):
                if (z_buf[j][2] < 0):
                    down.append(z_buf[j])
                elif(z_buf[j][2] >= 0):
                    up.append(z_buf[j])
                    
                    
                
    i = 1;
    while (i < len(up)- 1):
        if ((up[i][2] + up[i - 1][2]) / 2) > ((up[i + 1][2] + up[i - 1][2]) / 2) and ((up[i][2] + up[i + 1][2]) / 2) > ((up[i + 1][2] + up[i - 1][2]) / 2):
            up.pop(i)
            i = 1
        else:
            i += 1
    while (i < len(down)- 1):
        if ((down[i][2] + down[i - 1][2]) / 2) > ((down[i + 1][2] + down[i - 1][2]) / 2) and ((down[i][2] + down[i + 1][2]) / 2) > ((down[i + 1][2] + down[i - 1][2]) / 2):
            down.pop(i)
            i = 1
        else:
            i += 1
    down.reverse()
    new_list = up + down
    return new_list
def get_new_parts2_elements(part, percent):
    if (len(part[0]) == 1):
        return [[part[0][0], part[1][0], part[2][0]]]
    xyz = []
    copier = copy.deepcopy(part)
    for i in range(len(copier[0])):
        xyz.append([copier[0][i],copier[1][i], copier[2][i]])
    xyz.sort(key = lambda x:x[0])
    #print(xyz)
    cur_x = xyz[int(percent * (len(xyz) - 1) / 100)][0]
    ans = []
    if (int(percent * (len(xyz) - 1)) // 100 == 0):
        for i in xyz:
            if i[0] == cur_x:
                return [[i[0], i[1], -1], [i[0], i[1], 1]]
    if (int(percent * (len(xyz) - 1)) // 100 == len(xyz) - 1):
        for i in xyz:
            if i[0] == cur_x:
                return [[i[0], i[1], -1], [i[0], i[1], 1]]
    for i in xyz:
        if i[0] == cur_x:
            ans.append(i)
    if (len(ans) > 2):
        ans = [max(ans, key = lambda x: x[2]), min(ans, key = lambda x: x[2])]
    return ans
def get_new_parts2(parts):
    #print(parts[1])
    begin = copy.deepcopy(parts[1])
    len_begin = len(begin[0])
    slice_amount = 25
    new_parts = []
    #print(begin)
    for i in range(slice_amount):
        cur_ind = i * len_begin / slice_amount
        percent = cur_ind * 100 / len_begin
        one_slice = []
        print(percent)
        for j in range(len(parts)):
            slice_part = get_new_parts2_elements(parts[j], percent)
            if len(slice_part) > 0:
                one_slice = one_slice + slice_part
        if len(one_slice) > 0:
            one_slice = new_parts_sort2(one_slice)
            #new_parts = new_parts + one_slice
            new_parts.append(one_slice)
    percent = 100
    one_slice = []
    for j in range(len(parts)):
        one_slice = one_slice + get_new_parts2_elements(parts[j], 100)
    if len(one_slice) > 0:
            one_slice = new_parts_sort2(one_slice)
            #new_parts = new_parts + one_slice
            new_parts.append(one_slice)
    return new_parts
def plot_other_parts2(other_parts):
    #print(other_parts)
    x = []; y = []; z = [];
    for i in other_parts:
        #print(i)
        x.append(i[0])
        y.append(i[1])
        z.append(i[2])
    ax = plt.axes(projection ='3d')
    ax.plot(x, y, z, color = 'm')
    plt.show()
# В следующих 3х функциях работаем с новой СД для brain, по вертикали соединяем спиралью все ребра
def new_parts_sort(one_part):
    one_part.sort(key = get_1)
    one_part.reverse()
    buf = []
    for i in range(len(one_part) - 1):
        #print(i, len(one_part))
        y1 = one_part[i][1]
        y2 = one_part[i + 1][1]
        if(y1 == y2):
            if (one_part[i][2] < one_part[i + 1][2]):
                buf.append(one_part.pop(i))
        if (i >= len(one_part) - 2):
            break
    buf.reverse()
    new_list = one_part + buf
    return new_list
def get_new_parts1(parts):
    x_min = get_min(parts,0)
    x_max = get_max(parts,0)
    step = 50
    x_cur = x_min
    new_parts = []
    while (x_cur < x_max):
        one_part = []
        for i in range(len(parts)):
            for j in range(len(parts[i][0])):
                if (parts[i][0][j] == x_cur):
                    one_part.append([parts[i][0][j], parts[i][1][j], parts[i][2][j]])
        part = new_parts_sort(one_part)
        new_parts = new_parts + part
        x_cur += step
    return new_parts
def plot_other_parts1(other_parts):
    #print(other_parts)
    x = []; y = []; z = [];
    for i in other_parts:
        #print(i)
        x.append(i[0])
        y.append(i[1])
        z.append(i[2])
    ax = plt.axes(projection ='3d')
    ax.plot(x, y, z, color = 'm')
    plt.show()
                
def move_to_center(brain):
    for i in brain.keys():
        xc,yc = get_center(brain[i])
        for j in brain[i]:
            j[0] -= xc
            j[1] -= yc
def move_to_point(brain, px, py):
    for i in brain.keys():
        if (i == 'top'):
            for j in brain[i]:
                j[0] += px
                j[1] += py
        elif (i == 'side'):
            for j in brain[i]:
                j[0] += py
        else:
            for j in brain[i]:
                j[0] += px
            
        
def read_brain(file) :
    print("reading file", file)
    file = open(data_folder + file, 'r')
    brain_left = []
    brain_right = []
    ind = 0
    for line in file:
        if (ind >= 2):
            data = line.split()
            brain_left.append([int(data[0]), int(data[1])]); brain_right.append([int(data[2]), int(data[3])])
        ind += 1

    brain_right.reverse()
    ans = brain_left + brain_right
    ans.append(brain_left[0])
    ans.remove(ans[0])
    
    return ans




in_files =  ["top.txt", "front.txt", "front_minus_5.txt", "front_minus_4.txt", "front_minus_3.txt", "front_minus_2.txt", "front_minus_1.txt",
             "front_plus_1.txt", "front_plus_2.txt", "front_plus_3.txt", "front_plus_4.txt", "front_plus_5.txt","side.txt"]
brain = {'top' : [], 'front' : [], 'side' : [], 'front_plus_1' : [], 'front_plus_2' : [], 'front_plus_3' : [], 'front_plus_4' : [], 'front_plus_5' : []
         , 'front_minus_1' : [], 'front_minus_2' : [], 'front_minus_3' : [], 'front_minus_4' : [], 'front_minus_5' : []}
for i in range(len(in_files)):

    file = in_files[i]
    key = ''
    key += file[:-4]
    brain[key] = (read_brain( in_files[i]))

for i in brain.keys():
    print("w and h for ", i, " is ", get_wid(brain[i]), get_height(brain[i]), "len of ", i, " is ", len(brain[i]))
correction_sizes(brain)
move_to_center(brain)
move_to_point(brain, 590, 370)
parts = get_plots(brain)
closing_distance(parts)

parts_2d = make_2d_parts(parts)

other_parts2 = get_new_parts2(parts)

parts_sovokup = make_parts_sovokup(parts_2d, other_parts2)
X, Y, Z, neurs, Dots = extract_points(parts_sovokup, 10)
links_neur = get_neur_links(neurs)
links_neur = random.sample(links_neur, len(links_neur) // 256)
N_X = np.array(neurs[0]); N_Y = np.array(neurs[1]); N_Z = np.array(neurs[2])
'''
f = open("parts_dots.txt", 'w')
for i in range(len(X)):
    f.write(str(X[i]) + ' ' + str(Y[i]) + ' ' +str(Z[i]) + '\n')
f.close()'''

f = open(data_folder +"parts_dots.txt", 'w')
for i in range(len(Dots)):
    f.write(str(Dots[i][0]) + ' ' + str(Dots[i][1]) + ' ' +str(Dots[i][2]) + '\n')
f.close()

f = open(data_folder +"neurs_dots.txt", 'w')
for i in range(len(neurs[0])):
    f.write(str(neurs[0][i]) + ' ' + str(neurs[1][i]) + ' ' +str(neurs[2][i]) + '\n')
f.close()



fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z, s = 1, color = 'red')

fig.show()




