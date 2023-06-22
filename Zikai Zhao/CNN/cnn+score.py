# Import required modules
# 4.0:加入循环，分数
# !!!!!!!!!!!!!!!!!!!!!!open彩图得到rgb矩阵，但是本征脸得到灰度矩阵,所以本征脸用convert转化一下！！！！！！！！！！
import os

import cv2
import cv2 as cv
import time
import argparse

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import LabelEncoder



def loadDataProcessing(file):
    with open(file) as f:
        s = f.read().splitlines()  # 读取逐行数据放入list列表中
    allMessage = []
    for i in range(len(s)):
        # print(s[9])#打印第九行
        temp = s[i]
        temp = temp.split(' ')  # 按空格划分字符串为list列表
        while '' in temp:  # 删除list中的空元素
            temp.remove('')
        # print(temp)


    # 对单行进行赋值处理
        if len(temp) > 10:
            id = temp[0]
            sex = temp[2][:-1]
            age = temp[4][:-1]
            race = temp[6][:-1]
            face = temp[8][:-1]
            prop = temp[10:]
            prop[0] = prop[0][2:]



            prop.pop(-1)
            # 判断prop是否为空，并且根据prop特征的个数来用空格进行拼接
            if len(prop) != 0:
                str = ''
                for i in range(len(prop) - 1):
                    str = str + prop[i] + ' '
                str = str + prop[len(prop) - 1]
                prop = str
            else:
                prop = 'NaN'

        # print(prop)
            temp = [id, sex, age, race, face, prop]
            allMessage.append(temp)
        # print(temp)
        else:
            pass
        # id = temp[0]
        # temp = [id, "missing descriptor"]
        # print(temp)
        # allMessage.append(temp)
    return allMessage


def roadLabels(path):
    # 将字符串标签“male”，“female”处理为1，0
    if (type(path).__name__ == 'list'):  # 用于处理列表数据
        labels = []
        for i in range(len(y)):
            labels.append(y[i][1])
    else:  # 用于处理csv文件
        df = pd.read_csv(path)
        labels = df.iloc[:, 1].values.tolist()
    # for i in range(len(labels)):
    #     if labels[i] == "male":
    #         labels[i] = 0
    #     else:
    #         labels[i] = 1
    le = LabelEncoder()
    labels = le.fit_transform(labels)
    # print(labels)
    return labels

parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
parser.add_argument('--input',
                    help='Path to input image or video file. Skip this argument to capture frames from a camera.')

args = parser.parse_args()

faceProto = "./opencv_face_detector.pbtxt"
faceModel = "./opencv_face_detector_uint8.pb"

genderProto = "./gender_deploy.prototxt"
genderModel = "./gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

genderList = ['Male', 'Female']

# Load network

genderNet = cv.dnn.readNet(genderModel, genderProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)

# Open a video file or an image file or a camera stream

path="./pic"
files = os.listdir(path)

"保存预测的值"
ypred=[]

for i in range(1,3992):
    # print(file)
    # print(type(file))
    # !open彩图得到rgb矩阵，但是本征脸得到灰度矩阵,所以本征脸用convert转化一下！
    image = Image.open("./pic"+"/"+str(i)+".jpg").convert("RGB")
    # image = cv2.imread(file, cv2.IMREAD_COLOR)
    image = np.asarray(image)
    # print(len(image))
    # print(len(image[0]))

    # print(image[0][0])#第一个矩阵的第一个像素的rgb列表

    face=image
    frameOpencvDnn=face
    frameFace=frameOpencvDnn

    blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
    genderNet.setInput(blob)
    genderPreds = genderNet.forward()
    gender = genderList[genderPreds[0].argmax()]
    # print("Gender Output : {}".format(genderPreds))
    print("Gender : {}, conf = {:.3f}".format(gender, genderPreds[0].max()))
    label = "{}".format(gender)
    # print(gender)
    # print(type(gender))
    ypred.append(gender)

    cv.imshow("Age Gender Demo", frameFace)

le = LabelEncoder()
ypred = le.fit_transform(ypred)

# print(ypred)
y1 = loadDataProcessing('faceDR')
y2 = loadDataProcessing('faceDS')
y = y1 + y2
y = roadLabels(y)
y = np.delete(y, 1190)
y = np.delete(y, 1186)

a1=np.sum(ypred==y)/ypred.size
print(a1)
