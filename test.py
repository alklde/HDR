import torch
from DeepLearning import DeepLearning
from DataLoader import data_loader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from PIL import ImageOps
import os
import random

def convert_to_gray(img_path, black_bg=False):
    """ 转换图片灰度并进行中心裁剪 """
    with Image.open(img_path) as img:
        img = img.convert("L")

        # 转换为黑底白字
        if not black_bg:
            img = ImageOps.invert(img)

        # 中心裁剪
        arr = np.array(img)
        coords = np.argwhere(arr > 0)
        if coords.size == 0:
            raise ValueError("输入图片是空白的！")
        y0, x0 = coords.min(axis=0) # 左上角坐标
        y1, x1 = coords.max(axis=0) # 右下角坐标
        crop = arr[y0 : y1 + 1, x0 : x1 + 1] 

        # 保持纵横比缩放到 20x20
        h, w = crop.shape
        if h > w:
            new_h = 20
            new_w = int(round((w * 20.0) / h))
        else:
            new_w = 20
            new_h = int(round((h * 20.0) / w))

        if new_h == 0 or new_w == 0:
            return None            
        crop_img = Image.fromarray(crop).resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 填充到 28x28
        new_img = Image.new("L", (28, 28), 0)
        left = (28 - new_w) // 2
        top = (28 - new_h) // 2
        new_img.paste(crop_img, (left, top)) 

        new_img_array = np.array(new_img)
        new_img = torch.tensor(new_img_array, dtype=torch.float32) / 255.0
        new_img = new_img.unsqueeze(0).unsqueeze(0)
        return new_img
    
def text_dataset(path):
    """ 从文本图像中读取图片 """
    with open(path, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

        if not lines:
            raise ValueError("文件为空或只包含空白内容")
        # 检查是否所有行长度相同
        line_lengths = [len(line) for line in lines]
        if len(set(line_lengths)) > 1:
            raise ValueError("文件中包含不同长度的行")
        
        # 将字符串转换为整数列表
        arr = np.array([[int(c) for c in line] for line in lines], dtype=np.uint8)
        height, width = arr.shape
        return arr, height, width

def text_to_img(path):
    """ 预处理文本数据集 """
    arr, height, width = text_dataset(path)
    arr = arr * 255
    img = Image.fromarray(arr)
    out_path = path.replace(".txt", ".png")
    img.save(out_path)
    return out_path

def predict(model, img):
    """ 预测单张图片 """
    if len(img.shape) == 3:
        img = img.unsqueeze(0)
    outputs = model(img)
    pred = torch.argmax(outputs)
    return pred.item()



if __name__ == "__main__":
    model = DeepLearning()
    test_data = data_loader(is_train=False)
    model.load_state_dict(torch.load("model.pth", weights_only=True))

    # 随机显示
    # for (n, (x, _)) in enumerate(test_data):
    #     if n > 9:
    #         break
    #     img = x[0].view(-1, 28 * 28)
    #     outputs = model.forword(img)
    #     pred = torch.argmax(outputs)
    #     plt.subplot(2, 5, n+1)
    #     plt.imshow(x[0].view(28, 28), cmap='gray')
    #     plt.title(f"Pred: {pred.item()}")
    #     plt.axis('off')
    # plt.show()

    # # 预测自定义图片
    # while True:
    #     path = input("请输入图片地址: ")
    #     try:
    #         img = model.preprocess(Image.open(path))
    #         img_array = np.array(img)
    #         img = torch.tensor(img_array, dtype=torch.float32) / 255.0
    #         pred = predict(model, img)
    #         plt.imshow(img_array, cmap='gray')
    #         plt.title(f"Pred: {pred}")
    #         plt.axis('off')
    #         plt.show()
    #     except FileNotFoundError:
    #         print("文件不存在，请重新输入")

    # for i in range(10):
    #     path = f"data/MyUse/pred{i}.png"
    #     img = convert_to_gray(path, black_bg=True)
    #     pred = predict(model, img)
    #     plt.subplot(2, 5, i+1)
    #     plt.imshow(img.squeeze(), cmap='gray')
    #     plt.title(f"Pred: {pred}")
    #     plt.axis('off')
    # plt.show()

    path = "data/digits/testDigits"
    correct = 0
    total = 0
    for i in range(10):
        for j in range(100):
            text_path = f"{path}/{i}_{j}.txt"
            if not os.path.exists(text_path):
                break
            img_path = text_to_img(text_path)
            img = convert_to_gray(img_path, black_bg=True)
            if img is None:
                continue
            pred = predict(model, img)
            if pred == i:
                correct += 1
            total += 1
    print(f"Acc: {correct}/{total} ({correct/total:.4f})")

    count = 0
    while count < 10:
        ans = random.randint(0, 9)
        idx = random.randint(0, 100)
        text_path = f"{path}/{ans}_{idx}.txt"
        if not os.path.exists(text_path):
            continue
        img_path = text_to_img(text_path)
        img = convert_to_gray(img_path, black_bg=True)
        if img is None:
            continue
        pred = predict(model, img)
        count += 1
        plt.subplot(2, 5, count)
        plt.imshow(img.squeeze(), cmap='gray')
        plt.title(f"Pred: {pred}")
        plt.axis('off')
    plt.show()