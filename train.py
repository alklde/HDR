import torch
from DeepLearning import DeepLearning
from DataLoader import data_loader
import matplotlib.pyplot as plt
from tqdm import tqdm

def train(model, train_data, test_data, loss_fn, optimizer, epochs=5):
    """ 训练模型 """
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for (img, label) in tqdm(train_data, desc=f"Epoch {epoch + 1} / {epochs}"):
            outputs = model(img)
            loss = loss_fn(outputs, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc = model.evaluate(test_data)

        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Acc: {acc:.4f}")
    print(f"Best Acc: {best_acc:.4f}")

if __name__ == "__main__":
    model = DeepLearning()
    train_data = data_loader(is_train=True)
    test_data = data_loader(is_train=False)

    # 定义损失函数和优化器
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    train(model, train_data, test_data, loss_fn, optimizer, epochs=5)

    # 保存模型
    torch.save(model.state_dict(), "model.pth")