# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

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
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False)

    def forward(self, input):
        output = self.maxpool1(input)
        return output

tudui = Tudui()

writer = SummaryWriter("../logs/maxpool")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = tudui(imgs)
    writer.add_images("output", output, step)
    step = step + 1
    print(step)
writer.close()