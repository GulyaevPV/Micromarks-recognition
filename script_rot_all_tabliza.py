import sys
import os
import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy.ndimage.interpolation as ndii
import pprint
import skimage
import time
from scipy import ndimage, misc
from scipy.signal import argrelextrema
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
# global constants
RE_IDX = 0
IM_IDX = 1
ROWS_AXIS = 0
COLS_AXIS = 1
polarMode = "spline"
noiseMode = "none"  # "gaussian", "s&p", "none"
noiseIntensity = {'sigma': 2, 'mean': 0,
                  'whiteThreshold': 0.01, 'blackThreshold': 0.99}
resultsComparation = True


def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]


def ang(lineA, lineB):
    # Get nicer vector form
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
    # Get dot prod
    dot_prod = dot(vA, vB)
    # Get magnitudes
    magA = dot(vA, vA)**0.5+0.00001
    magB = dot(vB, vB)**0.5+0.00001
    # Get cosine value
    #cos_ = dot_prod/magA/magB
    # Get angle in radians and then convert to degrees
    angle = math.acos(dot_prod/magB/magA)
    # Basically doing angle <- angle mod 360
    ang_deg = math.degrees(angle) % 360
  #  print ("Anglessss = " + str(ang_deg))
    #print (ang_deg)
    if ang_deg-180 >= 0:
        # As in if statement
        return 360 - ang_deg
    else:

        return ang_deg


def load_images(name_file_A, name_file_B):
    A = cv2.imread(name_file_A, cv2.IMREAD_GRAYSCALE) > 128
    if np.sum(A) > (A.shape[0]*A.shape[1]/2):
        A = np.logical_not(A)

    B = cv2.imread(name_file_B, cv2.IMREAD_GRAYSCALE) > 128
    if np.sum(B) > (B.shape[0]*B.shape[1]/2):
        B = np.logical_not(B)
    return A, B


def distance(x1, y1, x2, y2):
    return ((x1 - x2)**2 + (y1 - y2)**2)**0.5


def get_iou(A, B):
    # Массивы A и B двумерные с размерностями [width, height] бинарные
    t, p = A, B
    true = np.sum(t)
    pred = np.sum(p)
    # Количество единичных пикселей в ground truth массиве A
    # и в предсказанном массиве B

    # Если маска нулевая (чтобы извежать деления 0/0)
    if true == 0:
        return pred == 0

    # Пересечение находится поэлементным перемножением массивов
    intersection = np.sum(t * p)
    # Объединение - сумма количества единиц в A и B минус пересечение
    union = true + pred - intersection
    iou = intersection / union
    return iou


def get_iou_vector(A, B):
    # Функция для набора (батча) изображений
    # Массивы A и B трехмерные с размерностями [batch_size, width, height]
    # Первая размерность - количество изображений в батче
    batch_size = A.shape[0]
    # Расчет средней метрики по набору изображений
    metric = 0.0
    for batch in range(batch_size):
        metric += get_iou(A[batch], B[batch])
    metric /= batch_size
    return metric

# main script


def main(argv):
    timeStart = time.time()


path_in = r'.\sho4'
files = os.listdir(path_in)
count = 0
X = []
Sc = []
for file_name in files:
    img = cv2.imread(path_in + '\\'+file_name)
    #img=ndimage.median_filter(img, size=5)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    #kernel=np.ones((3,3), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=3)
    #img = erosion

    # Convert the image to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # выравнивание освещенности
    clahe = cv2.createCLAHE(clipLimit=5)
    final_img = clahe.apply(gray) + 30
    cv2.imwrite('houghlines5.jpg', final_img)
    #if gray.std() < 17:
        #print(file_name)
    if gray.std() < 7:
        ret, thresh3 = cv2.threshold(gray, 70, 71, cv2.THRESH_TRUNC) #  # cv2.threshold(gray, 60, 61, cv2.THRESH_TRUNC)   cv2.threshold(gray,46,250,cv2.THRESH_BINARY) 
        thresh = skimage.filters.threshold_isodata(gray)
        #ret,thresh3 = cv2.threshold(gray,thresh,255,cv2.THRESH_BINARY)
        #gray=final_img
        gray = thresh3
        #gray=final_img
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        #gray = cv2.erode(gray, kernel, iterations=3)
        '''
        alpha = 2.5 # Contrast control (1.0-3.0)
        beta = 10 # Brightness control (0-100)
        adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        #gray=adjusted
        '''
    # адаптивный порог
    #gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,5,3)

    abs_e = np.absolute(gray)
    r8u1 = np.uint8(abs_e)
    # edges = cv2.Canny(r8u1, 450, 451,apertureSize = 5)
    edges = cv2.Canny(r8u1, 450, 451, apertureSize=5)
    cv2.imwrite('edges.jpg', edges)

    inverted_image = cv2.bitwise_not(edges)
    cv2.imwrite('inverted_image.jpg', gray)

    thr = [10, 20, 40, 60, 80]
    for th in thr:
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, th,
                                minLineLength=30, maxLineGap=50)  # maxLineGap=50
        if not(lines is None):
            if len(lines) < 35:
                break

    if (lines is None):
        lines = []

    for line in lines:  # все линии
        x1, y1, x2, y2 = line[0]
        cv2.line(erosion, (x1, y1), (x2, y2), (0, 0, 0), 3)

    cv2.imwrite('houghlines3.jpg', erosion)
    plt.subplot(221), plt.imshow(gray, cmap='gray')
    plt.show()

    height, width = img.shape[:2]
    h1 = height/7
    dlin = [280, 150, 100]
    for dlina in dlin:
        ar = []
        for line in lines:
            for line1 in lines:
                x1, y1, x2, y2 = line[0]
                lineA = ((x1, y1), (x2, y2))
                x3, y3, x4, y4 = line1[0]
                lineB = ((x3, y3), (x4, y4))
                # число верхняя граница длины отрезка
                if h1 < distance(x1, y1, x2, y2) < dlina:
                    if h1 < distance(x3, y3, x4, y4) < dlina:
                        g = ang(lineB, lineA)
                        rtol = 1000.1
                        # if not (np.allclose(line[0],line1[0], 0.1, 10.1)): #добавляем если массивы не идентичны
                        #if not (np.array_equal(line[0],line1[0])):
                        ar.append(g)

              #  g=ang(lineB, lineA)
              #  ar.append(g)
        t = plt.hist(ar)
        m = max(t[0])
        arr = t[0]
        arr2 = t[0]
        # добавляем пустой элемент вначале, что выделит локальный максимум вначале массива
        arr2 = np.insert(arr2, 0, 0)
        arr2 = np.append(arr2, 0)  # добавляем пустой элемент вконце
        arrr = t[1]
        arrr2 = t[1]
        # добавляем пустой элемент, чтобы выделить локальный максимум вначале массива
        arrr2 = np.insert(arrr2, 0, -10)
        # ищем индексы локальных максимумов
        max_ind = argrelextrema(arr2, np.greater_equal, mode='wrap')
        arrr2 = np.append(arrr2, -1)

        y = list(max_ind)
        # преобразуем в массив с которым можно работать
        for i in range(len(max_ind)):
            y[i] = y[i]+1
        max_ind1 = tuple(y)
    #  max_ind1 массив в котором индексы увеличены на 1 - то есть указание на конец диапазона  элемента гистограммы
        r1 = arrr2[max_ind]
        r2 = arrr2[max_ind1]
        # print(r)
        i = arr.argmax()  # индекс максимума
        # поиск максимумов гистограмм около 0 и 90
        flag = 0
        zerof = 0
        if 160 < arrr[i+1] < 190 or 160 < arrr[i] < 190:
            arr[i] = 0
        if -3 < arrr[i] < 10:  # для нулевого диапазона берем первое значение диапазон
            if arr[i] != 0:
                flag = flag+1
                zerof = 1
            arr[i] = 0
        i = arr.argmax()
        if 160 < arrr[i+1] < 190 or 160 < arrr[i] < 190:
            arr[i] = 0
            i = arr.argmax()
        # для диапазона 70-90 берем второе значение диапазон
        if 70 < arrr[i+1] < 110 or 70 < arrr[i] < 110:
            if arr[i] != 0:
                flag = flag+1
            arr[i] = 0

        i = arr.argmax()
        # проверка на случай, если главный максимум около 90
        if (-3 < arrr[i] < 10 and flag < 2 and zerof == 0) or (-70 < arrr[i+1] < 110 and flag < 2 and zerof == 0):
            if arr[i] != 0:
                flag = flag+1
        # второй критерий
        flag1 = 0
        zerof = 0
        f90 = 0
        r3 = arr2[max_ind]  # массив локальных максимумов
        y = list(max_ind1)
        r4 = r3
        # преобразуем в массив с которым можно работать
        r_all = 0
        for i in range(len(max_ind1)):

            if (i+1) < len(arr2):
                r4[i] = arr2[i+1]
        r_all = 0
        for i in range(len(arr2)):
            r_all = r_all+arr2[i]

        r5 = 0
        for i in range(len(r1)):
            if (-3 < r1[i] < 10 and zerof == 0 and (r3[i] != 0 or r4[i] != 0)) or (-3 < r2[i] < 10 and zerof == 0 and (r3[i] != 0 or r4[i] != 0)):
                flag1 = flag1+1
                zerof = 1
                r5 = r5+r3[i]
            if (80 < r1[i] < 100 and f90 == 0 and (r3[i] != 0 or r4[i] != 0)) or (80 < r2[i] < 100 and f90 == 0 and (r3[i] != 0 or r4[i] != 0)):  # было 75 104
                flag1 = flag1+1
                f90 = 1
                r5 = r5+r3[i]

        if flag == 2 or flag1 == 2:
            break

    plt.show()
    # t=plt.hist(ar)
    # IoU metrics
    #file_name_A = r"ugol_inv_mask2.png"
    #file_name_B = r"houghlines4.jpg"
    #A, B = load_images(file_name_A, file_name_B)
    # print("IoU=")
    #print (get_iou(A,B))
    '''
    print("flag=") 
    print (flag)
    print("flag1=") 
    print (flag1)
    '''
    if flag >= 2 or flag1 >= 2:
        print('1')
        print(r5/(r_all+1))
        print(file_name)
        X.append(1)
        Sc.append(r5/(r_all+1))
    else:
        print('0')
        print(r5/(r_all+1))
        #print(file_name)
        count = count+1
        X.append(0)
        Sc.append(r5/(r_all+1))
print(count)
X1 = []
for i in range(0, 111):
    if i > 52:
        X1.append(1)
    else:
        X1.append(0)
precision, recall, thresholds = precision_recall_curve(X1, Sc)
np.savetxt("foo.csv", recall, delimiter=",")
np.savetxt("foo1.csv", precision, delimiter=",")
disp = PrecisionRecallDisplay(precision=precision, recall=recall)
disp.plot()
print(average_precision_score(X1, Sc))
print(roc_auc_score(X1, Sc))
if __name__ == '__main__':
    sys.exit(main(sys.argv))
