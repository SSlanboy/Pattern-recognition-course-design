#文件读取及处理
import os
import numpy as np
from PIL import Image
#数据预处理
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import cv2
#分类器
from sklearn.neural_network import MLPClassifier

#测试验证
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report


##读取图像文件

folder_path = r"rawdata"
image_size = (128, 128)
#取1时分类性别，取2时分类年龄，取3时分类种族，取4时分类表情
classification_kind=1
# 读取图像文件
image_data = []
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'rb') as file:
        image_bytes = file.read()
        image_data.append(image_bytes)

# 转换为图像数据
images = []
for image_bytes in image_data:
    image = Image.frombytes('L', image_size, image_bytes)
    images.append(image)

# 将图像数据转换为像素矩阵
pixel_matrices = []
for image in images:
    pixels = np.array(image, dtype=np.uint8)
    pixels = np.array(pixels).flatten()
    pixel_matrices.append(pixels)

#PCA降维为100维的特征
pca = PCA(n_components=128)
pca_data = pca.fit_transform(pixel_matrices)

# scaler = StandardScaler()
# pca_data = scaler.fit_transform(pca_data)

lable=[]
#读取预先整理好的标签文件
with open("processed_data.txt", 'r') as input_file:
    for line in input_file:
        data = line.strip().split(',')
        lable.append(data[classification_kind])
encoder = LabelEncoder()
lable = encoder.fit_transform(lable)
#查看映射
# label_mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

X = pca_data
y = lable
#数据标准化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 使用k-fold交叉验证评估BP分类器准确性
BP = MLPClassifier(hidden_layer_sizes=(500,300), activation='relu', solver='lbfgs', random_state=42,max_iter=1000)
kfold = KFold(n_splits=10, shuffle=True, random_state=42)
BP_scores = cross_val_score(BP, X, y, cv=kfold, scoring='accuracy')

#随机分一个子集作为此时的评价标准
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
BP.fit(X_train, y_train)
y_pred = BP.predict(X_test)
report = classification_report(y_test, y_pred)   #生成此时的分类报告


print("Mean accuracy of BP: ", BP_scores.mean()) #打印kfold下的平均准确率
print("Classification report:")
print(report)






