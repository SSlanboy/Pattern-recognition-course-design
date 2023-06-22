from sklearn import svm
import os
import numpy as np
import cv2 as cv
import math
from skimage import feature as skif
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn import decomposition
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from lbp_basic import lbp_basic


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


def match_PicGen(path1, y1, y2):
    """
    将性别标签与图片像素矩阵相匹配
    并将字符串标签“male”，“female”处理为1，0：
    param
    path1：rawdata文件路径
    y1：dr文件路径
    y2：ds文件路径
    return
    s：图片像素矩阵列表
    g：性别标签列表
    """
    # 获取数据标签
    y1 = loadDataProcessing(y1)
    y2 = loadDataProcessing(y2)
    y = y1 + y2

    # 读取rawdata文件夹下的文件，返回像素矩阵列表和性别标签列表
    path = path1  # 文件夹目录
    files = os.listdir(path)  # 得到文件夹下的所有文件名称
    s = []
    # s.append('pixels')
    g = []
    # g.append('gender')
    for file in files:  # 遍历文件夹
        with open(path + '/' + file, 'rb') as f:
            x = np.fromfile(f, dtype=np.uint8)
            if len(x) == 128 * 128:  # 剔除像素不为128*128的图片
                for y_index in y:  # 遍历标签
                    if y_index[0] == file and y_index[1] != None:
                        x = x.reshape(128, 128)
                        #                         cv2.imshow("orgin", x)
                        #                         cv2.waitKey(0)
                        s.append(x)
                        if y_index[1] == 'male':
                            gen_lable = 0
                        else:
                            gen_lable = 1
                        g.append(gen_lable)

    return s, g


def mblbp():
    x, y = match_PicGen("./rawdata", 'faceDR', 'faceDS')  # 读取文件夹下的文件
    print('now is classificating')
    # print(x)
    # print(y)
    gendata = []
    for i in range(len(y)):
        for j in range(len(x)):
            if i == j:
                t = (y[i], x[j],)
                gendata.append(t)
    print(gendata[1250])

    # 特征提取(mb-LBP)————————————————————————————————————————————————————————————————

    g_feature = []
    g_lable = []
    print("extracting feature with LBP...")
    # 在tqdm()函数里检索可以获得运行进度
    for mat_i in tqdm(gendata):
        hist = lbp_basic(mat_i[1])  # (, 2, 8)
        #     print(hist)
        g_feature.append(hist)
        #     print(g_feature)
        lable = mat_i[0]
        g_lable.append(lable)
    print("extraction completed !")
    # print("extracting feature with BRINT...")
    # for img in tqdm(gendata):
    #     dst, hist = brint_s(img[1], 2, 32)
    #     g_feature.append(hist)
    #     lable = img[0]
    #     g_lable.append(lable)
    # print("extraction completed !")
    # print(g_feature)
    # print(g_lable)
    Gfeature = np.array(g_feature)
    print('Gfeature_shape:', Gfeature.shape)
    print('Gfeature[0]:', Gfeature[0])

    # re_Gfeature：后续可以直接处理的特征
    re_Gfeature = []
    for mat in Gfeature:
        res = np.array(mat).reshape(-1)
        re_Gfeature.append(res)
    # print(re_Gfeature)
    re_Gfeature = np.array(re_Gfeature)
    print('re_fea_shape:', re_Gfeature.shape)
    print('re_Gfeature[0]:', re_Gfeature[0])

    return re_Gfeature,g_lable


# x, y = mblbp()

# # PCA降维————————————————————————————————————————————————————————————————
# pca = decomposition.PCA()
# pca.fit(x)
# # 观察占比可以选择降多少维
# print(pca.explained_variance_ratio_)
# # X为降维后数据
# #X = pca.fit_transform(x)
# X = x
#
# stdsc = StandardScaler()  # 标准化
# x = stdsc.fit_transform(X)
# # # scaler = MinMaxScaler() # 归一化
# # # x = scaler.fit_transform(x)
# x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# or
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
# x_train=stdsc.fit_transform(x_train)
# x_test=stdsc.fit_transform(x_test)

#
# clf = svm.SVC()
# #clf = AdaBoostClassifier(n_estimators=50)
# clf.fit(x_train, y_train)
#
# y_hat = clf.predict(x_test)
# y_score = clf.decision_function(x_test)
#
# # -------------------------------------------------------------------------------
# print("混淆矩阵：")
# print(confusion_matrix(y_test, y_hat))
# print("准确率：", accuracy_score(y_test, y_hat))
# print("精确率：", precision_score(y_test, y_hat, pos_label=1, average='binary'))
# print("召回率：", recall_score(y_test, y_hat, pos_label=1, average='binary'))
# print("F1值：", f1_score(y_test, y_hat, pos_label=1, average='binary'))
#
# y = np.array(y_test)
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
