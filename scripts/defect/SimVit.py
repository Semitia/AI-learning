import os
import timm
import numpy as np
import torchvision,torch,time

from torch import nn
from vit_pytorch import SimpleViT
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential

# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
script_dir = os.path.dirname(os.path.abspath(__file__)) # 获取脚本所在的目录
os.chdir(script_dir) # 切换到脚本所在的目录
current_directory = os.getcwd()
print("Current working directory:", current_directory)
# 网络
# os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
writer_dir = '../../logs/SimpleVit/'
model_dir = '../../models/SimpleVit/'

data_dir = '../../dataset/Summer_camp/dataset1/'
train_data = datasets.ImageFolder(root=data_dir + 'train', transform=transforms.ToTensor())
test_data = datasets.ImageFolder(root=data_dir + 'val', transform=transforms.ToTensor())

# 计算均值和标准差
mean = np.zeros(3)
std = np.zeros(3)
num_images = 0
for img, _ in train_data:
    mean += img.mean(dim=[1, 2]).numpy()
    std += img.std(dim=[1, 2]).numpy()
    num_images += 1
mean /= num_images
std /= num_images
print(f'Mean: {mean}')
print(f'Std: {std}')
transform = transforms.Compose([
    # transforms.RandomResizedCrop(64),       # 随机裁剪并调整大小
    transforms.RandomHorizontalFlip(),        # 随机水平翻转
    transforms.ToTensor(),                    # 转换为张量
    transforms.Normalize(mean=mean, std=std)  # 归一化
])
train_data = datasets.ImageFolder(root=data_dir + 'train', transform=transform)
test_data = datasets.ImageFolder(root=data_dir + 'val', transform=transform)
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度为：{}".format(train_data_size))
print("测试数据集的长度为：{}".format(test_data_size))
# 打印类别到索引的映射
print("Class to index mapping:", train_data.class_to_idx)

tudui = SimpleViT(
    image_size = 64,
    patch_size = 8,
    num_classes = 9,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048
).to(device)

loss_fn = nn.CrossEntropyLoss() # 交叉熵损失函数
loss_fn = loss_fn.to(device)
learning_rate = 1e-3
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)
input = torch.randn(64, 3, 64, 64).to(device)
output = tudui(input)
print(output.shape)

total_train_step = 0    # 训练次数
total_test_step = 0     # 测试次数
epoch = 200              # 训练轮数
best_acc = 0
writer = SummaryWriter(writer_dir)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

last_time = time.time()
for i in range(epoch):
    print("-------第 {} 轮训练开始-------".format(i+1))
    # 训练
    tudui.train() # 对某些特定的层需要此句
    for data in train_dataloader:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = tudui(inputs)
        loss = loss_fn(outputs, labels)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        # if total_train_step % 100 == 0:
        #     print("训练次数：{}, Loss: {}".format(total_train_step, loss.item()))
        #     writer.add_scalar("train_loss", loss.item(), total_train_step)

    # 测试
    tudui.eval() # 对某些特定的层需要此句
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = tudui(inputs)
            loss = loss_fn(outputs, labels)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (outputs.argmax(1) == labels).sum() # argmax取最大值，(0)竖着看，(1)横着看
            total_accuracy = total_accuracy + accuracy

    total_accuracy = total_accuracy/test_data_size
    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_accuracy))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy, total_test_step)
    total_test_step = total_test_step + 1
    if total_accuracy > best_acc:
        # torch.save(v.state_dict(), "../models/res_sample/res_{}.pth".format(i))
        torch.save(tudui.state_dict(), model_dir + "/res_best.pth")
        best_acc = total_accuracy
    delta_time = time.time() - last_time
    last_time = time.time()
    print("模型已保存，耗时：{}".format(delta_time))

writer.close()

