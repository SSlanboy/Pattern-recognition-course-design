import os
import time
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torchvision.transforms import transforms
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
folder_path = r"rawdata"
image_size = (128, 128)

# 定义超参数
learning_rate = 0.0001
num_epochs = 100
patience = 5  # 早停阈值：连续达到准确率阈值的周期数
classification_kind = 2   # 取1时分类性别，取2时分类年龄，取3时分类种族，取4时分类表情
batch_size = 64
L2=0.1

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

class CustomDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        x = self.X[index]
        x = Image.fromarray(x, mode='L')
        if self.transform:
            x = self.transform(x)
        y = self.y[index]
        return x, y

# 创建自定义数据集对象
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # 将图像从单通道灰度转换为3通道灰度
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 创建自定义数据集对象
dataset = CustomDataset(pixel_matrices, labels, transform=transform)

# 定义交叉验证的折数（k）
k_folds = 3

# 创建StratifiedKFold对象
skf = StratifiedKFold(n_splits=k_folds, shuffle=True)

# 创建LeNet模型
class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.resnet = resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        return self.resnet(x)

# 初始化类别统计变量
class_total = [0] * num_label

# 统计各类别样本数量
for _, labels in DataLoader(dataset, batch_size=len(dataset), shuffle=False):
    for label in labels:
        class_total[label] += 1

# 创建文件夹保存模型和结果
save_dir = "classification_results"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
average_acc=[]
# 训练和测试模型
for fold, (train_indices, test_indices) in enumerate(skf.split(pixel_matrices, labels)):
    start_time = time.time()
    # 划分训练集和测试集
    train_set = torch.utils.data.Subset(dataset, train_indices)
    test_set = torch.utils.data.Subset(dataset, test_indices)

    # 创建数据加载器
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # 创建模型
    model = ResNet(num_classes=num_label).to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 定义早停变量
    best_epoch = 0
    best_accuracy = 0.0
    current_patience = 0

    # 训练和评估模型
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计训练损失和准确率
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_correct += (predicted == labels).sum().item()

        # 计算训练集的平均损失和准确率
        train_loss = train_loss / len(train_set)
        train_accuracy = train_correct / len(train_set)

        # 测试模型
        model.eval()
        test_loss = 0.0
        test_correct = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # 统计测试损失和准确率
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == labels).sum().item()

        # 计算测试集的平均损失和准确率
        test_loss = test_loss / len(test_set)
        test_accuracy = test_correct / len(test_set)

        # 输出训练和测试结果
        # print(f"Fold: {fold + 1}/{k_folds}, Epoch: {epoch + 1}/{num_epochs}, "
        #       f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
        #       f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # 保存最佳模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_dir, f"fold{fold + 1}_best_model.pt"))
            current_patience = 0
        else:
            current_patience += 1

        # 判断是否早停
        if current_patience == patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        # # 打印每个epoch的训练时间
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # print(f"Time elapsed for epoch {epoch + 1}: {elapsed_time:.2f} seconds")

    print(f"Best model for fold {fold + 1} was at epoch {best_epoch + 1} with accuracy {best_accuracy:.4f}")
    # 加载最佳模型进行测试
    model.load_state_dict(torch.load(os.path.join(save_dir, f"fold{fold + 1}_best_model.pt")))
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        predicted_labels = []
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predicted_labels.extend(predicted.cpu().tolist())

        accuracy = 100 * correct / total

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Training time: {total_time:.2f} seconds")
    # 创建类别标签列表
    class_labels = [i for i in range(num_label)]

# 创建混淆矩阵
    confusion_matrix = np.zeros((num_label, num_label), dtype=int)

# 统计每个类别的预测结果
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
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
    
    # 计算测试集的准确率、F1-score和ROC曲线
    test_predictions = []
    test_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_predictions.extend(predicted.tolist())
            test_labels.extend(labels.tolist())

    test_accuracy = sum(np.array(test_predictions) == np.array(test_labels)) / len(test_labels)
    test_f1_score = f1_score(test_labels, test_predictions, average='weighted')
    # fpr, tpr, _ = roc_curve(test_labels, test_predictions)
    # roc_auc = auc(fpr, tpr)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1-score: {test_f1_score:.4f}")
    # print(f"Test ROC AUC: {roc_auc:.4f}")
    # # 绘制ROC曲线
    # plt.figure()
    # plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title(f'ROC Curve (Fold {fold + 1})')
    # plt.legend(loc="lower right")
    # plt.savefig(os.path.join(save_dir, f"fold{fold + 1}_roc_curve.png"))
