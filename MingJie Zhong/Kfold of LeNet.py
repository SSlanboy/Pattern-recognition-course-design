import os
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

folder_path = r"rawdata"
image_size = (128, 128)

# 定义超参数
learning_rate = 0.001
num_epochs = 100
patience = 8  # 早停阈值：连续达到准确率阈值的周期数
classification_kind = 4   # 取1时分类性别，取2时分类年龄，取3时分类种族，取4时分类表情
L2=0.01

num_label_array = [2, 4, 5, 3]
str_label_array = ['sex', 'age', 'race', 'expression']
num_label = num_label_array[classification_kind - 1]


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
    pixel_matrices.append(pixels)

# 读取预先整理好的标签文件
labels = []
with open("encoded_data.txt", 'r') as input_file:
    for line in input_file:
        data = line.strip().split(',')
        labels.append(int(data[classification_kind]))

# 创建数据集
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        x = self.X[index]
        x = torch.from_numpy(x).unsqueeze(0).float()  # 转换为张量并添加通道维度
        y = self.y[index]
        return x, y

# 创建自定义数据集对象
dataset = CustomDataset(pixel_matrices, labels)

# 定义交叉验证的折数（k）
k_folds = 5

# 创建StratifiedKFold对象
skf = StratifiedKFold(n_splits=k_folds, shuffle=True)

# 创建LeNet模型
class LeNet(nn.Module):
    def __init__(self, label):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 29 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, label)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 初始化类别统计变量
class_total = [0] * num_label

# 统计各类别样本数量
for _, labels in DataLoader(dataset, batch_size=len(dataset)):
    for label in labels:
        class_total[label.item()] += 1

# 计算类别权重
class_weights = torch.tensor([1 / class_total[i] for i in range(num_label)])

start_time = time.time()

# 定义交叉验证的训练和测试过程
def train_model(model, train_loader, test_loader, criterion, optimizer, patience):
    model.train()
    best_accuracy = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for i, (images, labels) in enumerate(train_loader):
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss /= len(train_loader)
        accuracy = 100 * correct / total

        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')  # 保存最佳模型
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} as accuracy has not improved for {patience} epochs.")
            break

    return best_accuracy


# 执行k折交叉验证
fold = 0
total_accuracy = 0.0

for train_index, test_index in skf.split(pixel_matrices, labels):
    fold += 1
    print(f"Fold {fold}:")

    train_data = [pixel_matrices[i] for i in train_index]
    train_labels = [labels[i] for i in train_index]
    test_data = [pixel_matrices[i] for i in test_index]
    test_labels = [labels[i] for i in test_index]

    # 创建训练集和测试集的自定义数据集对象
    train_dataset = CustomDataset(train_data, train_labels)
    test_dataset = CustomDataset(test_data, test_labels)

    # 创建数据加载器
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 创建LeNet模型
    model = LeNet(label=num_label)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=L2)

    # 初始化损失函数时传入类别权重
    criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(weight=class_weights)   #执行L2正则化

    # 训练模型
    best_accuracy = train_model(model, train_loader, test_loader, criterion, optimizer, patience)
    print(f"Best accuracy for fold {fold}: {best_accuracy:.2f}%")
    total_accuracy += best_accuracy

average_accuracy = total_accuracy / fold
print(f"Average accuracy: {average_accuracy:.2f}%")

# 加载最佳模型的状态字典
model.load_state_dict(torch.load('best_model.pth'))

# 测试模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    f1 = f1_score(labels, predicted, average='weighted')


end_time = time.time()
total_time = end_time - start_time

print(f"Test Accuracy: {accuracy:.2f}%, Test F1: {f1:.4f}")
print(f"Training time: {total_time:.2f} seconds")
print(f"Training class: {str_label_array[classification_kind-1]}")

# 创建类别标签列表
class_labels = [i for i in range(num_label)]

# 创建混淆矩阵
confusion_matrix = np.zeros((num_label, num_label), dtype=int)

# 统计每个类别的预测结果
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        for i in range(len(predicted)):
            true_label = labels[i]
            pred_label = predicted[i]
            confusion_matrix[true_label][pred_label] += 1

# 打印每个类别的分类情况
print("Classification Results:")
for i in range(num_label):
    class_name = class_labels[i]
    correct = confusion_matrix[i][i]
    total = np.sum(confusion_matrix[i])
    accuracy = correct / total * 100
    print(f"Class: {class_name}, Correct: {correct}, Total: {total}, Accuracy: {accuracy:.2f}%")


#绘制auc曲线

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# 获取每个类别的预测概率
probabilities = []
true_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        probabilities.extend(outputs.numpy())
        true_labels.extend(labels.numpy())

probabilities = np.array(probabilities)
true_labels = np.array(true_labels)

# 计算每个类别的假阳性率和真阳性率
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(num_label):
    fpr[i], tpr[i], _ = roc_curve((true_labels == i).astype(int), probabilities[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# 绘制AUC曲线
plt.figure()
colors = ['red', 'green', 'blue', 'orange', 'purple']  # 可根据类别数量调整颜色列表
for i in range(num_label):
    plt.plot(fpr[i], tpr[i], color=colors[i], label='Class %s (AUC = %0.2f)' % (i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()