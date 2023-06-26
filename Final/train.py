import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
from dataset import FishDataset
from torchvision.transforms import ToTensor

model = fasterrcnn_resnet50_fpn(pretrained=True)

train_path = '/home/student/Desktop/class/class_master1_bot/interaction/Final/FinalProj-FishDetection/TraingData/'

# 建立模型實例
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
# 定義損失函數和優化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 創建訓練資料集
train_dataset = FishDataset(train_path)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
loss_fn = torch.nn.CrossEntropyLoss()
transform = ToTensor()

# 訓練模型
model.train()
for images, targets in train_loader:
    # 將圖像和目標轉換為張量
    # images = [transform(image) for image in images]
    images = [image for image in images]
    targets = [{k: v for k, v in target.items()} for target in targets]

    # 清除之前計算的梯度
    optimizer.zero_grad()

    # 執行前向傳播
    predictions = model(images, targets)

    # 計算損失
    loss = sum(loss_fn(prediction, target) for prediction, target in zip(predictions, targets))

    # 執行反向傳播和優化
    loss.backward()
    optimizer.step()
