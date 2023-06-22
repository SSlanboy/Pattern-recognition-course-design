import numpy as np
from sklearn.preprocessing import LabelEncoder

# 读取数据标签文件
data_labels = np.load('data_lable.npy', allow_pickle=True)

# 获取特征列的编码结果和特征种类数
num_features = data_labels.shape[1] - 1
encoded_labels = []
num_classes = []

for i in range(num_features):
    label_encoder = LabelEncoder()
    encoded_feature = label_encoder.fit_transform(data_labels[:, i+1])
    encoded_labels.append(encoded_feature)
    num_classes.append(len(label_encoder.classes_))

# 将编码后的特征和图像名称重新组合成数组
encoded_labels = np.column_stack(encoded_labels)
encoded_data_labels = np.column_stack((data_labels[:, 0], encoded_labels))

# 转换为整数类型
encoded_data_labels = encoded_data_labels.astype(np.int64)

# 保存到文本文件

output_file_path = "encoded_data.txt"

with open(output_file_path, 'w') as output_file:
    for row in encoded_data_labels:
        row_str = ','.join(map(str, row))
        output_file.write(row_str + '\n')

print("Encoded data saved to 'encoded_data.txt'.")


# 打印保存完成的提示
print("Encoded data labels saved to 'encoded_data_labels.txt'.")
