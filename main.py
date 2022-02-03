from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QGraphicsScene
from PyQt5.QtGui import QPen, QImage, QPixmap, qRgb, qRgba, QPainter, QColor, QBrush, QPainterPath
from PyQt5.QtCore import Qt, QPoint, QTimer
from math import sin, cos, exp, sqrt
import numpy as np
from math import pi, sin, cos, fabs
import time
import copy
from threading import Thread
import random
data_folder = "data\\"
EPS = 1e-9
WIDTH, HEIGHT  = 1181, 741

#
#
#
#
# В move для impulse измени так, чтобы не было замыканий где if (self.sign < EPS):
#
#

def get_len(dot1, dot2):
    return sqrt((dot2.x - dot1.x)**2 + (dot2.y - dot1.y)**2 + (dot2.z - dot1.z)**2)

def sign_(x):
    if x >= 0:
        return 1
    else:
        return -1
class Dot:
    def __init__(self, x_, y_, z_):
        self.x = x_
        self.y = y_
        self.z = z_

    def __eq__(self, other):
        return bool (self.x == other.x and self.y == other.y and self.z == other.z)
        
    def rotateX(self, teta):
        teta = teta * pi / 180
        y_temp = self.y;
        z_temp = self.z;

        self.y = y_temp * cos(teta) - z_temp * sin(teta);
        self.z = y_temp * sin(teta) + z_temp * cos(teta);


    def rotateY(self, teta):
        teta = teta * pi / 180
        x_temp = self.x;
        z_temp = self.z;


        self.x = x_temp * cos(teta) + z_temp * sin(teta);
        self.z = -x_temp * sin(teta) + z_temp * cos(teta);


    def rotateZ(self, teta):
        teta = teta * pi / 180
        x_temp = self.x;
        y_temp = self.y;


        self.x = x_temp * cos(teta) - y_temp * sin(teta);
        self.y = x_temp * sin(teta) + y_temp * cos(teta);
    
    
    def transform(self, tetax, tetay, tetaz, center):

        self.x -= center[0]; self.y -= center[1]; self.z -= center[2];
        self.rotateX(tetax)
        self.x += center[0]; self.y += center[1]; self.z += center[2];

        self.x -= center[0]; self.y -= center[1]; self.z -= center[2];
        self.rotateY(tetay)
        self.x += center[0]; self.y += center[1]; self.z += center[2];

        self.x -= center[0]; self.y -= center[1]; self.z -= center[2];
        self.rotateZ(tetaz)
        self.x += center[0]; self.y += center[1]; self.z += center[2];
        
        self.x, self.y, self.z = round(self.x), round(self.y), round(self.z)        


class Impulse:
    def __init__(self, neur_, from_, to_, delta_):
        self.impulse = copy.deepcopy(from_)
        self.neur = neur_
        self.direction = [copy.deepcopy(from_), copy.deepcopy(to_)]
        self.step_del = max (   abs(from_.x - to_.x),
                                abs(from_.y - to_.y),
                                abs(from_.z - to_.z))
        
        self.c_x = 3 * (self.neur.cur_par1.x - from_.x)
        self.c_y = 3 * (self.neur.cur_par1.y - from_.y)
        self.c_z = 3 * (self.neur.cur_par1.z - from_.z)
        
        self.b_x = 3 * (self.neur.cur_par2.x - self.neur.cur_par1.x)- self.c_x
        self.b_y = 3 * (self.neur.cur_par2.y - self.neur.cur_par1.y)- self.c_y
        self.b_z = 3 * (self.neur.cur_par2.z - self.neur.cur_par1.z)- self.c_z

        self.a_x = to_.x - from_.x - self.c_x - self.b_x
        self.a_y = to_.y - from_.y - self.c_y - self.b_y
        self.a_z = to_.z - from_.z - self.c_z - self.b_z

        self.speed = 2.0
        #self.t = self.speed / 1000#self.step_del
        self.delta = delta_ * self.speed
        self.t = 0.0
        self.to_kill = 0
        self.sign = 1.0
        self.passed_route = 0.0

    def move(self, neur_):

            #self.impulse.x = self.a_x * pow(self.t,3) + self.b_x * pow(self.t,2) + self.c_x * self.t + self.direction[0].x;
            #self.impulse.y = self.a_y * pow(self.t,3) + self.b_y * pow(self.t,2) + self.c_y * self.t + self.direction[0].y
            #self.impulse.z = self.a_z * pow(self.t,3) + self.b_z * pow(self.t,2) + self.c_z * self.t + self.direction[0].z
            #print("T IS ", self.t)
            tmp_dot = copy.deepcopy(self.impulse)
            self.impulse.x =    self.direction[0].x * pow(1 - self.t,3) +\
                                3 * pow(1 - self.t,2) * self.t * neur_.cur_par1.x + \
                                3 * (1 - self.t) * pow(self.t,2) * neur_.cur_par2.x + \
                                pow(self.t,3) * self.direction[1].x

            self.impulse.y =    self.direction[0].y * pow(1 - self.t,3) + \
                                3 * pow(1 - self.t,2) * self.t * neur_.cur_par1.y + \
                                3 * (1 - self.t) * pow(self.t,2) * neur_.cur_par2.y + \
                                pow(self.t,3) * self.direction[1].y
            self.impulse.z =    self.direction[0].z * (1 - self.t)**3 + \
                                3 * (1 - self.t)**2 * self.t * neur_.cur_par1.z + \
                                3 * (1 - self.t) * self.t**2 * neur_.cur_par2.z + \
                                self.t**3 * self.direction[1].z
            #self.direction[0].x = self.impulse.x
            #self.direction[0].y = self.impulse.y
            #self.direction[0].z = self.impulse.z
            self.passed_route += get_len(tmp_dot, self.impulse)
            self.t += (self.delta * self.sign)
            #РАЗОБРАТЬСЯ С ЭТИМ ЯВНО, ЧТОБЫ НЕ БЫЛО ЗАМЫКАНИЙ НЕЙРОНОВ, А ТО КАК ИНСУЛЬТ ТУДА СЮДА ТУДА СЮДА
            if self.t >= 1:
                self.to_kill = 1
                print("KILLED")
                #self.to_kill = 1
# ТУТ ТИПА ПОПАЕМ ИМПУЛЬС ИЗ НЕЙРОНА, НАМ ПОНАДОБИТСЯ ПОЗЖЕ ДЛЯ ОСВОБОЖДЕНИЯ ПАМЯТ
            #if (self.to_kill == 1):
            #    print("KILLED ", self.neur.d1.x, self.neur.d1.y, self.neur.d1.z,"---", self.neur.d2.x, self.neur.d2.y, self.neur.d2.z,)
            #    self.neur.impulses.pop(self.neur.impulses.index(self))

    def transform(self, tetax, tetay, tetaz, center):
        self.impulse.transform(tetax, tetay, tetaz, center)
        
        self.direction[0].transform(tetax, tetay, tetaz, center)
        self.direction[1].transform(tetax, tetay, tetaz, center)
        
        
        '''self.vect = [self.direction[1].x - self.direction[0].x, self.direction[1].y - self.direction[0].y, self.direction[1].z - self.direction[0].z]
        
        self.step_del = max (   abs(self.direction[0].x - self.direction[1].x),
                                abs(self.direction[0].y - self.direction[1].y),
                                abs(self.direction[0].z - self.direction[1].z)
                            )
        self.t = 1 / self.step_del'''
    #def __del__(self):
        #print("deleting")
        #print(".")

class Neur:
    def __init__(self, d1_, cp_1_, cp_2_, d2_):
        self.d1 = copy.deepcopy(d1_)
        self.d2 = copy.deepcopy(d2_)
        self.cur_par1 = cp_1_
        self.cur_par2 = cp_2_
        self.impulse_amount = 0
        self.impulses = []
        self.neibours = []
        self.len = 0.0
    def generate_impulse(self):
        path = QPainterPath()
        path.moveTo(self.d1.x, HEIGHT - self.d1.y)
        path.cubicTo(self.cur_par1.x, self.cur_par1.y, self.cur_par2.x, self.cur_par2.y, self.d2.x, HEIGHT - self.d2.y)
        delta = 1 / path.length()
        self.len = path.length()
        self.impulses.append(Impulse(self, self.d1, self.d2, delta))
        self.impulse_amount += 1
    def transform(self, tetax, tetay, tetaz, center):
        self.d1.transform(tetax, tetay, tetaz, center)
        self.d2.transform(tetax, tetay, tetaz, center)
        self.cur_par1.transform(tetax, tetay, tetaz, center)
        self.cur_par2.transform(tetax, tetay, tetaz, center)
    

def get_parts_dots(file):
    parts = []
    f = open(data_folder + file, 'r')
    for line in f:
            one_part = []
            data = line.split()
            one_part.append(int(data[0]))
            one_part.append(int(data[1]))
            one_part.append(int(data[2]))
            parts.append(one_part)
    return parts
'''def get_neurs(file):
    parts = []
    f = open(file, 'r')
    flag = 0
    for line in f:
            one_part = []
            data = line.split()
            if (flag == 1):
                Dot_to = Dot(int(data[0]), int(data[1]), int(data[2]))
                curve_param_1 = Dot(random.randint(340, 750), random.randint(100, 650), int(data[2]))
                curve_param_2 = Dot(random.randint(340, 750), random.randint(100, 650), int(data[2]))
                Neur_ = Neur(Dot_from, curve_param_1, curve_param_2, Dot_to)
                #Neur_.generate_impulse()
                parts.append(Neur_)
            Dot_from = Dot(int(data[0]), int(data[1]), int(data[2]))
            flag = 1
    parts = random.sample(parts, len(parts) // 2)
    for i in parts:
        i.generate_impulse()
    return parts'''
def get_neurs(file):
    parts = []
    f = open(data_folder + file, 'r')
    flag = 0
    for line in f:
            one_part = []
            data = line.split()
            if (flag == 1):
                Dot_to = Dot(int(data[0]), int(data[1]), int(data[2]))
                
                parts.append(Dot_from);parts.append(Dot_to)
                
            Dot_from = Dot(int(data[0]), int(data[1]), int(data[2]))
            flag = 1
    
    neurs = []
    i = 0
    parts = random.sample(parts, len(parts) // 6)
    while i < len(parts):
        branches = random.randint(2, 4)
        Dot_from = parts[i]
        parts.pop(parts.index(Dot_from))
        if len(parts) >= branches:
            copy_parts = copy.deepcopy(parts)
            for j in range(branches):
                min_len = 999999999
                for k in range(len(copy_parts)):
                    if get_len(Dot_from, copy_parts[k]) < min_len:
                        min_len = get_len(Dot_from, copy_parts[k])
                        min_Dot = copy_parts[k]
                copy_parts.pop(copy_parts.index(min_Dot))
                #Dot_to = random.choice(parts)
                curve_param_1 = Dot(random.randint(340, 750), random.randint(100, 650), random.choice(parts).z)
                curve_param_2 = Dot(random.randint(340, 750), random.randint(100, 650), random.choice(parts).z)
                Neur_ = Neur(Dot_from, curve_param_1, curve_param_2, min_Dot)
                neurs.append(Neur_)
        i += 1
                    

    #neurs = random.sample(neurs, len(neurs) // 1)
    add_neibours(neurs)
    for i in neurs:
        i.generate_impulse()
    return neurs
def add_neibours(neurs):
    for i in range(len(neurs)):
        for j in range(len(neurs)):
            if (i != j):
                if (neurs[i].d2 == neurs[j].d1):
                    neurs[i].neibours.append(neurs[j])
def get_other_parts(file):
    parts = []
    f = open(data_folder + file, 'r')
    for line in f:
            one_part = []
            data = line.split()
            one_part.append(int(data[0]))
            one_part.append(int(data[1]))
            one_part.append(int(data[2]))
            parts.append(one_part)
    return parts

def okrug(fig):
    if (fig > 0.0):
        sign = 1
    elif (fig < 0.0):
        sign = -1
    elif (fabs(fig) < EPS):
        sign = 0
        return 0.0
    drob_ch = fabs(fig) - abs(floor(fabs(fig)))
    if (drob_ch > 0.5 or fabs(drob_ch - 0.5) <= EPS):
        ans = ceil(fabs(fig))
    elif (drob_ch < 0.5):
        ans = floor(fabs(fig))
    ans *= sign
    return ans
def get_center(parts):
    x_left = get_min(parts, 0)
    x_right = get_max(parts, 0)

    y_left = get_min(parts, 1)
    y_right = get_max(parts, 1)

    z_left = get_min(parts, 2)
    z_right = get_max(parts, 2)

    center = []
    center.append((x_left + x_right) // 2); center.append((y_left + y_right) // 2); center.append((z_left + z_right) // 2)
    print(center)
    return center
def find_y(parts, x, z):
    for i in parts:
        for j in range(len(i[0])):
            if (i[0][j] == x or i[2][j] == z):
                print("compare ", i[0][j], " with ", x, " and ",  i[2][j], " with ", z)
            if (i[0][j] == x and i[2][j] == z):
                return i[1][j];
    print("didnt found")
def get_inds(parts, x, y, z):
    for i in range(len(parts)):
        for j in range(len(parts[i][0])):
            if (parts[i][0][j] == x and parts[i][2][j] == z and parts[i][1][j] == y):
                return i,j;
def sign(x):
    if not x:
        return 0
    else:
        return x / abs(x)

def diff_analizator(x0, y0, xn, yn, image):
    if (abs(x0 - xn) == 0 and abs(y0 - yn) == 0):

        image.setPixel(int(x0), image.height() - int(y0), qRgba(0, 255, 0,192))
        #draw_pix(okrug(x0), okrug(y0), color_update(COLOR, 1))
    else:
        if (fabs(x0 - xn) > fabs(y0 - yn)):
            leng = fabs(x0 - xn)
        else:
            leng = fabs(y0 - yn)
        dx = (xn - x0) / leng
        dy = (yn - y0) / leng
        i = 0
        xi = x0
        yi = y0
        while (i < leng):

            image.setPixel(int(xi), image.height() - int(yi), qRgba(0, 255, 0,192))
            #draw_pix(okrug(xi), okrug(yi), color_update(COLOR, 1))
            xi += dx
            yi += dy
            i += 1

M = 1#48
shx = 1181 / 2 + 50
shy = 741 / 2 - 50


def rotateX(x, y, z, teta):
    teta = teta * pi / 180
    y_temp = y;
    z_temp = z;

    y = y_temp * cos(teta) - z_temp * sin(teta);
    z = y_temp * sin(teta) + z_temp * cos(teta);
    return x, y, z


def rotateY(x, y, z, teta):
    teta = teta * pi / 180
    x_temp = x;
    z_temp = z;


    x = x_temp * cos(teta) + z_temp * sin(teta);
    z = -x_temp * sin(teta) + z_temp * cos(teta);
    return x, y, z


def rotateZ(x, y, z, teta):
    teta = teta * pi / 180
    x_temp = x;
    y_temp = y;


    x = x_temp * cos(teta) - y_temp * sin(teta);
    y = x_temp * sin(teta) + y_temp * cos(teta);
    return x, y, z


def tranform(x, y, z, tetax, tetay, tetaz, center):
    x -= center[0]; y -= center[1]; z -= center[2];
    x, y, z = rotateX(x, y, z, tetax)
    x += center[0]; y += center[1]; z += center[2];

    x -= center[0]; y -= center[1]; z -= center[2];
    x, y, z = rotateY(x, y, z, tetay)
    x += center[0]; y += center[1]; z += center[2];

    x -= center[0]; y -= center[1]; z -= center[2];
    x, y, z = rotateZ(x, y, z, tetaz)
    x += center[0]; y += center[1]; z += center[2];
    
    #x = x * M + shx
    #y = y * M + shy
    return round(x), round(y), round(z)
def other_horizon_dots(scene_width, scene_hight,
                  tx, ty, tz, image, parts, neurs, center):
    x_right = -1
    y_right = -1
    x_left = -1
    y_left = -1

    # инициализация массивов горизонтов
    painter = QPainter(image)
    #painter.setRenderHint(QPainter.Antialiasing)
    path = QPainterPath()
    #qRgba(0, 255, 0,192))))
    for i in range(len(neurs)):

                
                #neurs[i].d1.x, neurs[i].d1.y, neurs[i].d1.z = tranform(neurs[i].d1.x, neurs[i].d1.y, neurs[i].d1.z, tx, ty, tz, center)
                #neurs[i].d2.x, neurs[i].d2.y, neurs[i].d2.z = tranform(neurs[i].d2.x, neurs[i].d2.y, neurs[i].d2.z, tx, ty, tz, center)

                neurs[i].transform(tx, ty, tz, center)
                
                #painter.drawLine(neurs[i].d1.x, scene_hight - neurs[i].d1.y, neurs[i].d2.x, scene_hight -neurs[i].d2.y)
                path.moveTo(neurs[i].d1.x, scene_hight - neurs[i].d1.y)
                path.cubicTo(neurs[i].cur_par1.x, scene_hight - neurs[i].cur_par1.y, neurs[i].cur_par2.x, scene_hight -neurs[i].cur_par2.y, neurs[i].d2.x, scene_hight - neurs[i].d2.y)
                painter.setPen(QPen(QColor(qRgba(0, 198, 31,255))))#painter.setBrush(QBrush(QColor(qRgba(0, 198, 31,255))))
                painter.drawPath(path)
                #for j in neurs[i].impulses:
                    
                    #j.impulse.x, j.impulse.y, j.impulse.z = tranform(j.impulse.x, j.impulse.y, j.impulse.z, tx, ty, tz, center)
                    #j.delta = (1 / path.length())
                    #j.transform(tx, ty, tz, center)
                    #painter.setBrush(QBrush(QColor(qRgba(255, 223, 0, 128))))
                    #painter.drawEllipse(QPoint(j.impulse.x, scene_hight - j.impulse.y), 5, 5)
                    #image.setPixel(j.impulse.x, scene_hight - j.impulse.y, qRgba(255, 223, 0, 255))
    for i in range(len(neurs)):

                painter.setBrush(QBrush(QColor(qRgba(0, 198, 31,255))))
                painter.drawEllipse(QPoint(neurs[i].d1.x, scene_hight - neurs[i].d1.y), 4, 4)
                painter.drawEllipse(QPoint(neurs[i].d2.x, scene_hight - neurs[i].d2.y), 4, 4)
                for j in neurs[i].impulses:
                    
                    #j.impulse.x, j.impulse.y, j.impulse.z = tranform(j.impulse.x, j.impulse.y, j.impulse.z, tx, ty, tz, center)

                    j.transform(tx, ty, tz, center)
                    painter.setBrush(QBrush(QColor(qRgba(255, 223, 0, 128))))
                    painter.drawEllipse(QPoint(j.impulse.x, scene_hight - j.impulse.y), 4, 4)
                    #image.setPixel(j.impulse.x, scene_hight - j.impulse.y, qRgba(255, 223, 0, 255))
    # инициализация переменных
    x_right = -1
    y_right = -1
    x_left = -1
    y_left = -1

    # инициализация массивов горизонтов
    for i in range(len(parts)):
                parts[i][0], parts[i][1], parts[i][2] = tranform(parts[i][0], parts[i][1], parts[i][2], tx, ty, tz, center)
                if x_left != -1:
                    image.setPixel(x_left, image.height() - y_left, qRgba(128, 128, 128, 255))#qRgba(128, 128, 128, 255))
                x_left = parts[i][0]
                y_left = parts[i][1]
    return image



red = Qt.red
blue = Qt.blue
black = Qt.black
white = Qt.white

def get_parts(file):
    parts = []
    one_part = []
    one_part.append([])
    one_part.append([])
    one_part.append([])
    n = 0
    f = open(data_folder + file, 'r')
    for line in f:
        if n == 0:
            if (len(one_part[0]) > 0):
                parts.append(one_part)
            n = line.split()
            n = int(n[0])
            one_part = []
            one_part.append([])
            one_part.append([])
            one_part.append([])
        else:
            data = line.split()
            one_part[0].append(int(data[0]))
            #print(one_part[1])
            one_part[1].append(int(data[1]))
            one_part[2].append(int(data[2]))
            n -= 1
    return parts
def get_0(elem):
    return elem[0]
def get_1(elem):
    return elem[1]
def get_2(elem):
    return elem[2]
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
class Window(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        uic.loadUi("window.ui", self)
        self.scene = QGraphicsScene(0, 0, 1181, 741)
        self.scene.win = self
        self.vel = 60
        self.view.setScene(self.scene)
        self.image = QImage(1181, 741, QImage.Format_ARGB32)
        self.image.fill(Qt.red)
        self.pen = QPen(Qt.red)
        self.draw.clicked.connect(lambda: draw(self))
        self.dial_x.valueChanged.connect(lambda: draw(self))
        self.dial_y.valueChanged.connect(lambda: draw(self))
        self.dial_z.valueChanged.connect(lambda: draw(self))
        self.parts = get_parts("brain_.txt")
        self.neurs = get_neurs("neurs_dots.txt")
        self.parts_dots = get_parts_dots("parts_dots.txt")
        self.center = get_center(self.parts)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_)
        self.timer.start(1000 / self.vel)
        
    def move_impulses(self):
            for i in range(len(self.neurs)):
                j = 0
                while j < self.neurs[i].impulse_amount:
                        if ( self.neurs[i].impulses[j].to_kill == 1):
                            self.neurs[i].impulses.pop()
                            self.neurs[i].impulse_amount -= 1
                            for k in range(len(self.neurs[i].neibours)):
                                self.neurs[i].neibours[k].generate_impulse()
                            
                        else:
                            self.neurs[i].impulses[j].move(self.neurs[i])
                        j += 1
                    
    def update_(self):
        self.scene.clear()
        self.image.fill(Qt.black)
        tx = self.dial_x.value()
        ty = self.dial_y.value()
        tz = self.dial_z.value()
        


        copy_parts_dots = copy.deepcopy(self.parts_dots)
        
        copy_neurs = copy.deepcopy(self.neurs)
        
        self.image = other_horizon_dots(self.scene.width(), self.scene.height(),
                      tx, ty, tz, self.image, copy_parts_dots, copy_neurs, self.center)
        self.move_impulses()
        
        
        pix = QPixmap()
        pix.convertFromImage(self.image)
        self.scene.addPixmap(pix)

        




def draw(win):
    win.scene.clear()
    win.image.fill(Qt.black)
    tx = win.dial_x.value()
    ty = win.dial_y.value()
    tz = win.dial_z.value()
    


    copy_parts_dots = copy.deepcopy(win.parts_dots)
    
    copy_neurs = copy.deepcopy(win.neurs)
    
    #win.move_impulses()
    win.image = other_horizon_dots(win.scene.width(), win.scene.height(),
                  tx, ty, tz, win.image, copy_parts_dots, copy_neurs, win.center)

    
    
    pix = QPixmap()
    pix.convertFromImage(win.image)
    win.scene.addPixmap(pix)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())
