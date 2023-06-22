import pandas as pd
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn import svm, neighbors
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
# import matplotlib.pyplot as plt
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
            # print(x.shape)
            # print(x)

            # 重新reshape图片
            # x = x.reshape(-1, 128, 128, 1)
            # x = (x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4]))
            # 查看reshape后的图片shape
            # print(x.shape)

            # x = x.reshape(128, 128)
            # print((x))
            # print(np.shape((x)))

            # print(x[3])
            x = np.int_(x)

            # print(len(x))
            # print(type(x))
            # print(x)
            # print(x[3])
            s.append(x)
            # print(s)

    return s


x = roadData("./rawdata")  # 读取文件夹下的文件

# prat_2使用处理好的文件标签
# y1 = roadLabels("faceDR_cleared.csv")
# y2 = roadLabels("faceDS_cleared.csv")
# y = np.append(y1, y2)

# part_3使用原始数据标签
y1 = loadDataProcessing('faceDR')
y2 = loadDataProcessing('faceDS')
y = y1 + y2

# print(y)
y = roadLabels(y)

# print(y)
# print(len(x))
# print(len(y))
print(len(x[1186]))
print(len(x[1190]))
x.pop(1190)
x.pop(1186)
y = np.delete(y, 1190)
y = np.delete(y, 1186)
# y.pop(1190)
# y.pop(1186)
print(len(x))
print(len(y))

print(x)
print(type(x))

# pca降维，5-》2
# sklearn_pca = PCA(n_components=0.95)
# x = sklearn_pca.fit_transform(x)

# 网上4-》2
# x = (x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3] * x.shape[4]))
# y = (y.reshape(y.shape[0], y.shape[1] * y.shape[2] * y.shape[3] * y.shape[4]))

stdsc = StandardScaler()  # 标准化
x = stdsc.fit_transform(x)
# # scaler = MinMaxScaler() # 归一化
# # x = scaler.fit_transform(x)
#
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
# x_train=stdsc.fit_transform(x_train)
# x_test=stdsc.fit_transform(x_test)


# clf = svm.SVC()
# clf.fit(x_train, y_train)
# y_hat = clf.predict(x_test)
# y_score = clf.decision_function(x_test)
# print(y_score)


knn1 = neighbors.KNeighborsClassifier(n_neighbors=3)
knn1.fit(x_train, y_train)
y_pred1 = knn1.predict(x_test)
a1 = np.sum(y_pred1 == y_test) / y_pred1.size
print("准确率：", a1)
print("F1值：", f1_score(y_test, y_pred1, pos_label=1, average='binary'))

# print("混淆矩阵：")
# print(confusion_matrix(y_test, y_hat))
# print("准确率：", accuracy_score(y_test, y_hat))
# print("精确率：", precision_score(y_test, y_hat, pos_label=1, average='binary'))
# print("召回率：", recall_score(y_test, y_hat, pos_label=1, average='binary'))
# print("F1值：", f1_score(y_test, y_hat, pos_label=1, average='binary'))

y = np.array(y_test)
# print(y)
# print("AUC值：", roc_auc_score(y, y_score))
# fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
# plt.figure(figsize=(5, 5))
# plt.plot(fpr, tpr, color='darkorange',
#          label='ROC curve (area = %0.2f)' % roc_auc_score(y, y_score))  ###假正率为横坐标，真正率为纵坐标做曲线
# plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic curve')
# plt.legend(loc="lower right")
# plt.show()
