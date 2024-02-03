import torch
import torch.nn as nn
import torch.nn.functional as F
from core.data.custom_data import InputData, OutputData


class Block(nn.Module):
    def __init__(self, in_chnl, out_chnl, num):
        super(Block, self).__init__()
        self.seq = nn.Sequential()
        for i in range(num):
            if i == 0:
                self.seq.append(nn.Conv2d(in_chnl, out_chnl, 3, 1, 1))
            else:
                self.seq.append(nn.Conv2d(out_chnl, out_chnl, 3, 1, 1))
            self.seq.append(nn.ReLU(True))
        self.seq.append(nn.MaxPool2d(3, 2, 1))

    def forward(self, x):
        return self.seq(x)


class VGG16(nn.Module):
    def __init__(self, opt) -> None:
        super(VGG16, self).__init__()
        self.feature = nn.Sequential(
            Block(3, 64, 2),
            Block(64, 128, 2),
            Block(128, 256, 3),
            Block(256, 512, 3),
            Block(512, 512, 3)
        )

        self.classfier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(),
            nn.Linear(7 * 7 * 512, 2048),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, opt.num_classes),
            nn.Softmax(1)
        )
    
    def forward(self, data: InputData) -> OutputData:
        x = data.input
        y = self.feature(x)
        y = self.classfier(y)
        return OutputData(y)


if __name__ == '__main__':
    net = VGG16(10)
    x = torch.rand((1, 3, 224, 224))
    y = net(x)
    torch.onnx.export(net, x, "vgg16.onnx", opset_version=11,
                      input_names=['input'], output_names=['output'])
    print(y)

