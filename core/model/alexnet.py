import torch
from torch import nn
from core.data.custom_data import InputData, OutputData


class AlexNet(nn.Module):
    def __init__(self, opt):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 48, 11, 4, 2), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(True),
            nn.MaxPool2d(3, 2), # kernel_size, stride
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(48, 128, 5, 1, 2),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2),
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(128, 192, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(192, 192, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(192, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(3, 2)
        )
         # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(True),
            nn.Linear(2048, opt.num_classes),
        )

    def forward(self, data: InputData) -> OutputData:
        img = data.input
        feature = self.conv(img)
        output = self.fc(torch.flatten(feature, start_dim=1))
        return OutputData(output)


if __name__ == '__main__':
    net = AlexNet(5)
    x = torch.rand((1, 3, 224, 224))
    y = net(x)
    print(y)
