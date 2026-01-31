import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST

def data_loader(is_train=True):
    """ 加载MNIST数据集 """
    # 定义标准化
    to_tensor = transforms.Compose([transforms.ToTensor()])
    # 加载数据集
    dataset = MNIST("./data", train=is_train, download=True, transform=to_tensor)
    # 为训练集和测试集创建 DataLoader
    dl = DataLoader(dataset, batch_size=16, shuffle=True)
    return dl