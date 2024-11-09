from torch import nn
from myTrain import MyTrain
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(4096, 64),
            Linear(64, 9)
        )

    def forward(self, x):
        x = self.model1(x)
        return x

data_dir = './dataset/Summer_camp/dataset1/'
writer_dir = './logs/defect/'
model_dir = './models/defect/'
m = Tudui()

trainer = MyTrain(m, data_dir, model_dir, writer_dir)
trainer.train(epochs=300)
    