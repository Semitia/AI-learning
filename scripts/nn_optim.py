import torch, torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.tensorboard import SummaryWriter

# 加载FashionMNIST数据集
import os
script_dir = os.path.dirname(os.path.abspath(__file__)) # 获取脚本所在的目录
os.chdir(script_dir) # 切换到脚本所在的目录
print("Current working directory:", os.getcwd())
dataset = torchvision.datasets.FashionMNIST(root="../dataset", train=False, download=False, 
                                            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]))
dataloader = DataLoader(dataset, batch_size=64)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(1, 32, 5, padding=4),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

loss = nn.CrossEntropyLoss()
tudui = Tudui()
optimizer = torch.optim.SGD(tudui.parameters(), lr=0.01)
for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        inputs, labels = data
        optimizer.zero_grad()
        outputs = tudui(inputs)
        l = loss(outputs, labels)
        l.backward()
        optimizer.step()
        running_loss += l.item()
    print("epoch:", epoch, "loss:", running_loss)



