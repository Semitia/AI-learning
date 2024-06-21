# -*- coding: utf-8 -*-
# 作者：小土堆
# 公众号：土堆碎念
import torchvision
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print('false', vgg16_false)

vgg16_true = torchvision.models.vgg16(pretrained=True)
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print('true', vgg16_true)

train_data = torchvision.datasets.CIFAR10('../data', train=True, transform=torchvision.transforms.ToTensor(),
                                            download=True)




