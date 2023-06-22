import re
import numpy as np

#需要分别整合faceDR和faceDS到processed_data里面后再开始分类

file_path = r"faceDR"
lines_to_read = 2000  # 指定要读取的行数

# 用于存储整理后的二维数组
image_attributes = []

# 默认值
default_age = -1
default_race = ""
default_expression = ""

# 读取文本文件的指定行数
with open(file_path, 'r') as file:
    for i in range(lines_to_read):
        line = file.readline()
        if not line:
            break  # 文件读取完毕
        # 使用正则表达式提取属性
        image_number = re.findall(r'(\d+)', line)[0]
        sex = 1 if 'female' in line else 0
        age_match = re.findall(r'\(_age\s+(\w+)\)', line)
        age = age_match[0] if age_match else default_age
        race_match = re.findall(r'\(_race\s+(\w+)\)', line)
        race = race_match[0] if race_match else default_race
        expression_match = re.findall(r'\(_face\s+(\w+)\)', line)
        expression = expression_match[0] if expression_match else default_expression

        # 将属性添加到二维数组
        image_attributes.append([image_number, sex, age, race, expression])

        

# 保存数组到文件
output_file_path = "processed_data.txt"
with open(output_file_path, 'w') as output_file:
    for row in image_attributes:
        output_file.write(','.join(map(str, row)) + '\n')

# 在需要使用时读取文件
# 示例：逐行读取并打印数据
with open(output_file_path, 'r') as input_file:
    for line in input_file:
        data = line.strip().split(',')
np.save('data.npy', data)
loaded_data = np.load('data.npy')
print(len(loaded_data))
