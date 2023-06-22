import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
from skimage import data

img = data.coffee()


def lbp_basic(img):
    basic_array = np.zeros(img.shape, np.uint8)
    for i in range(basic_array.shape[0] - 2):
        for j in range(basic_array.shape[1] - 2):
            basic_array[i, j] = bin_to_decimal(cal_basic_lbp(img, i, j))
    return basic_array


def cal_basic_lbp(img, i, j):  # 比中心像素大的点赋值为1，比中心像素小的赋值为0，返回得到的二进制序列
    aver = 0
    for m in range(i - 2, i + 3):
        for n in range(j - 2, j + 3):
            aver += (img[m, n])
    aver /= 25
    sum = []
    # print("平均值为：%f",aver)

    if img[i - 1, j] > aver:
        sum.append(1)
    else:
        sum.append(0)
    if img[i - 1, j + 1] > aver:
        sum.append(1)
    else:
        sum.append(0)
    if img[i, j + 1] > aver:
        sum.append(1)
    else:
        sum.append(0)
    if img[i + 1, j + 1] > aver:
        sum.append(1)
    else:
        sum.append(0)
    if img[i + 1, j] > aver:
        sum.append(1)
    else:
        sum.append(0)
    if img[i + 1, j - 1] > aver:
        sum.append(1)
    else:
        sum.append(0)
    if img[i, j - 1] > aver:
        sum.append(1)
    else:
        sum.append(0)
    if img[i - 1, j - 1] > aver:
        sum.append(1)
    else:
        sum.append(0)
    return sum


def bin_to_decimal(bin):  # 二进制转十进制
    res = 0
    bit_num = 0  # 左移位数
    for i in bin[::-1]:
        res += i << bit_num  # 左移n位相当于乘以2的n次方
        bit_num += 1
    return res


def show_basic_hist(a):  # 画mblbp的直方图
    hist = cv.calcHist([a], [0], None, [256], [0, 256])
    hist = cv.normalize(hist, hist)
    plt.figure(figsize=(8, 4))
    plt.plot(hist, color='r')
    plt.xlim([0, 256])
    plt.show()



# img1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# basic_array = lbp_basic(img1)
# show_basic_hist(basic_array)
# plt.figure(figsize=(11, 11))
# plt.subplot(1, 2, 1)
# plt.imshow(img1)
# plt.subplot(1, 2, 2)
# plt.imshow(basic_array, cmap='Greys_r')
# plt.show()