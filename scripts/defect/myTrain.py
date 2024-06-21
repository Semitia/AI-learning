import os
import time
import torch
import numpy as np
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from vit_pytorch import SimpleViT
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


class MyTrain:
    def __init__(self, model, data_dir, model_dir, writer_dir, device=None, set_lr=1e-3, clash=False):
        # 设备配置
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using {self.device} device.")
        # script_dir = os.path.dirname(os.path.abspath(__file__)) # 获取脚本所在的目录
        # os.chdir(script_dir) # 切换到脚本所在的目录
        current_directory = os.getcwd()
        print("Current working directory:", current_directory)
        # 网络端口
        if clash:
            os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
            os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'

        self.data_dir = data_dir
        self.model_dir = model_dir
        self.writer_dir = writer_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.train_data, self.test_data, self.mean, self.std = self.data_pretreat()
        self.train_dataloader = DataLoader(self.train_data, batch_size=64)
        self.test_dataloader = DataLoader(self.test_data, batch_size=64)
        self.num_classes = len(self.train_data.classes)     

        self.model = model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=set_lr)
        
        self.writer = SummaryWriter(self.writer_dir)
        self.total_train_step = 0
        self.total_test_step = 0
        self.best_acc = 0
        
    def load_data(self):
        train_data = datasets.ImageFolder(root=os.path.join(self.data_dir, 'train'), transform=transforms.ToTensor())
        test_data = datasets.ImageFolder(root=os.path.join(self.data_dir, 'val'), transform=transforms.ToTensor())
        return train_data, test_data
    
    def data_pretreat(self):
        train_data, test_data = self.load_data()
                
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
        
        # 数据增强
        transform = transforms.Compose([
            # transforms.RandomResizedCrop(64),       # 随机裁剪并调整大小
            transforms.RandomHorizontalFlip(),        # 随机水平翻转
            transforms.ToTensor(),                    # 转换为张量
            transforms.Normalize(mean=mean, std=std)  # 归一化
        ])
        train_data.transform = transform
        test_data.transform = transform
        
        train_data_size = len(train_data)
        test_data_size = len(test_data)
        print("训练数据集的长度为：{}".format(train_data_size))
        print("测试数据集的长度为：{}".format(test_data_size))
        print("Class to index mapping:", train_data.class_to_idx) # 类别到索引的映射
        
        return train_data, test_data, mean, std

    def train(self, epochs):
        last_time = time.time()
        for epoch in range(epochs):
            print(f"------- Epoch {epoch + 1} Training Start -------")
            self.model.train()
            for inputs, labels in self.train_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.total_train_step += 1

            self.evaluate(epoch)
            delta_time = time.time() - last_time
            last_time = time.time()
            print(f"Model saved, Time taken: {delta_time}")

        self.writer.close()

    def evaluate(self, epoch):
        self.model.eval()
        total_test_loss = 0
        total_accuracy = 0
        y_true = []         # 真实标签
        y_scores = []        # 预测标签

        with torch.no_grad():
            for inputs, labels in self.test_dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, labels)
                total_test_loss += loss.item()
                total_accuracy += (outputs.argmax(1) == labels).sum().item()

                y_true.extend(labels.cpu().numpy())  
                y_scores.extend(outputs.softmax(dim=1).cpu().numpy()) 

        total_accuracy /= len(self.test_data)
        print(f"Overall test loss: {total_test_loss}")
        print(f"Overall test accuracy: {total_accuracy}")
        self.writer.add_scalar("test_loss", total_test_loss, self.total_test_step)
        self.writer.add_scalar("test_accuracy", total_accuracy, self.total_test_step)

        # 处理AUC和ROC
        y_true = label_binarize(y_true, classes=list(range(self.num_classes)))  # 假设类别数为num_classes
        y_scores = np.array(y_scores)
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_scores[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            self.writer.add_scalar(f"AUC_class_{i}", roc_auc[i], self.total_test_step)
        print("AUC per class:", roc_auc)

        self.total_test_step += 1
        if total_accuracy > self.best_acc:
            torch.save(self.model.state_dict(), os.path.join(self.model_dir, "res_best.pth"))
            self.best_acc = total_accuracy


if __name__ == "__main__":
    data_dir = '../dataset/Summer_camp/dataset1/'
    writer_dir = '../logs/SimpleVit/'
    model_dir = '../models/SimpleVit/'
    m = SimpleViT(
        image_size = 64,
        patch_size = 8,
        num_classes = 9,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048
    )

    trainer = MyTrain(m, data_dir, model_dir, writer_dir)
    trainer.train(epochs=200)
    
