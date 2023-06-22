import glob
import os
import numpy as np
import pandas as pd
from skimage.feature import hog
import cv2
from sklearn import neighbors
from sklearn.model_selection import train_test_split
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
    if type(path).__name__ == 'list':  # 用于处理列表数据
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


def hog_s(tpath):
    path = tpath  # 文件夹目录
    files = os.listdir(path)
    s = []
    for file in files:  # 遍历文件夹
        with open(path + '/' + file, 'rb') as f:
            i = np.fromfile(f, dtype=np.uint16)
            # i.reshape(2048,12294)

            img = cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2GRAY)
            normalised_blocks, x = hog(img, orientations=12, pixels_per_cell=(8, 8), cells_per_block=(4, 4),
                                       block_norm='L2-Hys', visualize=True)

        x = x.reshape(131072, )
        x = np.int_(x)
        # print(x)
        s.append(x)
    return s


x = (hog_s("./pic"))

y1 = loadDataProcessing('faceDR')
y2 = loadDataProcessing('faceDS')
y = y1 + y2

# print(y)
y = roadLabels(y)
y = np.delete(y, 1190)
y = np.delete(y, 1186)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

knn1 = neighbors.KNeighborsClassifier(n_neighbors=3)
knn1.fit(x_train, y_train)
y_pred1 = knn1.predict(x_test)
a1 = np.sum(y_pred1 == y_test) / y_pred1.size
print(a1)
