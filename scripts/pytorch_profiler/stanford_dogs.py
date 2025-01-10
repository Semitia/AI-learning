import os
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

# 检查 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据预处理
data_dir = '/dataset/StanfordDogs/images'
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图片大小为 224x224
    transforms.ToTensor(),  # 转为 Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 加载数据
dataset = ImageFolder(root=data_dir, transform=transform)
train_size = int(0.8 * len(dataset)) # 数据集划分（80% 训练，20% 验证）
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
class_names = dataset.classes  # 120 个品种的类名
# print(f"Classes: {class_names}")

# 加载预训练的 ResNet-50 模型
import torch.nn as nn
from torchvision.models import resnet50
model = resnet50(pretrained=True)# 加载预训练的 ResNet-50
model.fc = nn.Linear(model.fc.in_features, 120)# 替换全连接层 (120 个类别)
model = model.to(device)# 移动模型到 GPU

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

import torch
from tqdm import tqdm
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # 包装 DataLoader
    loader = tqdm(loader, desc="Training", leave=False)
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 更新进度条显示
        loader.set_postfix({
            'Loss': f'{loss.item():.4f}', 
            'Accuracy': f'{100.0 * correct / total:.2f}%'
        })
    accuracy = 100.0 * correct / total
    return running_loss / len(loader), accuracy

def validate(model, loader, criterion, device):
    """
    验证模型
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    accuracy = 100.0 * correct / total
    return running_loss / len(loader), accuracy

# 训练模型
num_epochs = 1
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
