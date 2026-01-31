import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

class DeepLearning(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 定义卷积层
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 定义池化层
        self.pool = torch.nn.MaxPool2d(kernel_size=2, padding=0)
        # 定义全连接层
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 128)
        self.fc2 = torch.nn.Linear(128, 10)

        self.relu = torch.nn.ReLU() # 激活函数

    def forward(self, x):
        """ 前向传播 """
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def evaluate(self, test_data):
        """ 评估模型在测试数据集上的准确率 """
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for (img, label) in tqdm(test_data, desc="预测中: "):
                outputs = self(img)
                for i, output in enumerate(outputs):
                    if torch.argmax(output) == label[i]:
                        correct += 1
                    total += 1
        return correct / total
    
class DataSet(torch.utils.data.Dataset):
    """ 自定义数据集类 """
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, label = self.data[idx]
        if self.transform:
            img = self.transform(img)
        return img, label