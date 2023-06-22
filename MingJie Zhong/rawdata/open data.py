import re

file_path = r"E:\deskpot\pr\课程项目\人脸图像识别\人脸图像识别\face\faceDR"

data = []

with open(file_path, 'r') as file:
    for line in file:
        line = line.strip()
        # 提取标签及特征
        label = line.split()[0]
        features = re.findall(r'\((.*?)\)', line)
        
        # 处理特征部分
        feature_dict = {}
        for feature in features:
            match = re.search(r'_([^ ]+) ([^)]+)', feature)
            if match:
                feature_name = match.group(1)
                feature_value = match.group(2)
                feature_dict[feature_name] = feature_value
        
        # 存储为二维数组
        data.append([label, feature_dict])

# 打印二维数组
for row in data:
    print(row)
