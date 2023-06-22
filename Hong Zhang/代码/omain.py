import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# import dataProcessing as dP
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



def roadData(path):
    # 读取文件夹下的文件，返回一个list列表
    path = path  # 文件夹目录
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    # f = open(path+"/"+files[0],'rb')
    # s=np.fromfile(f,dtype=np.ubyte)
    # s=np.int_(s)
    # flag=0
    s = []
    for file in files:  # 遍历文件夹
        with open(path + '/' + file, 'rb') as f:
            x = np.fromfile(f, dtype=np.uint8)

            # print(x[3])
            x = np.int_(x)
            # print(x)
            # print(x[3])
            s.append(x)

    return s


x = roadData("./face/rawdata")  # 读取文件夹下的文件

# prat_2使用处理好的文件标签
# y1 = roadLabels("faceDR_cleared.csv")
# y2 = roadLabels("faceDS_cleared.csv")
# y = np.append(y1, y2)

# part_3使用原始数据标签
y1 = loadDataProcessing('./face/faceDR')
y2 = loadDataProcessing('./face/faceDS')
y = y1 + y2

print(y)
y = roadLabels(y)

# print(y)
# print(len(x))
# print(len(y))
print(len(x[1186]))
print(len(x[1190]))
x.pop(1190)
x.pop(1186)
y = np.delete(y, 1190)#错误数据
y = np.delete(y, 1186)
# y.pop(1190)
# y.pop(1186)
print(len(x))
print(len(y))



#################################################以下可以直接移植自己的分类器


'''
变量说明:
    x:图片数据
    y:标签
'''



#pca降维
clf = PCA(n_components=0.95)
data_2 = clf.fit_transform(x)

#降维后的标准化
scaler = StandardScaler()
X1 = scaler.fit_transform(data_2)

#保存
np.savetxt(r"./feature1.csv", X1, delimiter=',')