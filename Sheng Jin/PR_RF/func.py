import os
import numpy as np
from tqdm import tqdm
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
    # x, y = match_PicGen("./rawdata_simple", 'faceDR', 'faceDS')  # 读取文件夹下的文件
    print('now is classificating')
    # print(x)
    # print(y)
    gendata = []
    for i in range(len(y)):
        for j in range(len(x)):
            if i == j:
                t = (y[i], x[j],)
                gendata.append(t)


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

