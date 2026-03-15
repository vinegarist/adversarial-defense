from unicodedata import name
import torch
import torch.nn as nn


class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        # 全连接层：28 * 28 -> 2（图像 -> 两类）
        self.features = nn.Linear(28*28, 1)
        # 使用sigmoid函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 归一化
        mean = torch.tensor(0.1215)
        std = torch.tensor(0.3011)
        x = (x - mean) / std
        
        x = self.features(x)
        x = self.sigmoid(x)
        x = x.squeeze(-1)
        return x


class LR10(nn.Module):
    def __init__(self):
        super(LR10, self).__init__()
        # 图片大小为28*28=784像素，类别为0~9共10类
        self.fc = nn.Linear(28*28, 10)

    def forward(self, x):
        # 归一化
        mean = torch.tensor(0.1307)
        std = torch.tensor(0.3081)
        x = (x - mean) / std
        
        x = self.fc(x)
        return x


class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(28*28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.reshape((x.shape[0], -1))
        # 归一化
        mean = torch.tensor(0.1307)
        std = torch.tensor(0.3081)
        x = (x - mean) / std

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 6, 5, 1, 2,),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5, 1, 0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        # 归一化
        mean = torch.tensor(0.1307)
        std = torch.tensor(0.3081)
        x = (x - mean) / std
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
