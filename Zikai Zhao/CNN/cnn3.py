# Import required modules
# 3.0:加入循环


# !!!!!!!!!!!!!!!!!!!!!!open彩图得到rgb矩阵，但是本征脸得到灰度矩阵,所以本征脸用convert转化一下！！！！！！！！！！
import os

import cv2
import cv2 as cv
import time
import argparse

import numpy as np
from PIL import Image


def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()#说明读的一帧图片的格式是矩阵列表
    frameHeight = frameOpencvDnn.shape[0]#矩阵行数
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, bboxes


parser = argparse.ArgumentParser(description='Use this script to run age and gender recognition using OpenCV.')
parser.add_argument('--input',
                    help='Path to input image or video file. Skip this argument to capture frames from a camera.')

args = parser.parse_args()

faceProto = "./opencv_face_detector.pbtxt"
faceModel = "./opencv_face_detector_uint8.pb"



genderProto = "./gender_deploy.prototxt"
genderModel = "./gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load network

genderNet = cv.dnn.readNet(genderModel, genderProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)

# Open a video file or an image file or a camera stream

path="./pic"
files = os.listdir(path)

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
    cv.imshow("Age Gender Demo", frameFace)
