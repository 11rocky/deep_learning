import torch
import torch.nn as nn
import torch.nn.functional as F
from core.data.custom_data import InputData, OutputData


class BasicConv2d(nn.Module):
    def __init__(self, in_chnls, out_chnls, kernel_size, stride=1, padding=0, **kwargs) -> None:
        super(BasicConv2d, self).__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(in_chnls, out_chnls, kernel_size, stride, padding, **kwargs),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.sequence(x)


class Inception(nn.Module):
    def __init__(self, in_chnls, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj) -> None:
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_chnls, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_chnls, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_chnls, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_chnls, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        return torch.cat([b1, b2, b3, b4], 1)


class InceptionAux(nn.Module):
    def __init__(self, in_chnls, num_classes) -> None:
        super(InceptionAux, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = BasicConv2d(in_chnls, 128, kernel_size=1)
        self.linear1 = nn.Linear(2048, 1024)
        self.linear2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.avgpool(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = F.dropout(x, 0.5, training=self.training)
        x = F.relu(self.linear1(x), True)
        x = F.dropout(x, 0.5, training=self.training)
        return self.linear2(x)


class GoogleNet(nn.Module):
    def __init__(self, opt) -> None:
        super(GoogleNet, self).__init__()
        self.aux_logits = opt.aux_logits
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
    
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)
        if self.aux_logits:
            self.aux1 = InceptionAux(512, opt.num_classes)
            self.aux2 = InceptionAux(528, opt.num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.linear = nn.Linear(1024, opt.num_classes)


    def forward(self, data: InputData) -> OutputData:
        x = data.input
        x = self.maxpool1(self.conv1(x))
        x = self.maxpool2(self.conv3(self.conv2(x)))
        x = self.maxpool3(self.inception3b(self.inception3a(x)))

        x = self.inception4a(x)
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)
        x = self.inception4d(self.inception4c(self.inception4b(x)))
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)
        x = self.maxpool4(self.inception4e(x))

        x = self.inception5b(self.inception5a(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.linear(x)
        if self.training and self.aux_logits:
            return OutputData(predict=x, aux_logits2=aux2, aux_logits1=aux1)
        else:
            return OutputData(predict=x)


if __name__ == "__main__":
    net = GoogleNet(10)
    x = torch.rand((1, 3, 224, 224))
    net.eval()
    y = net(x)
    torch.onnx.export(net, x, "googlenet.onnx", opset_version=11,
                      input_names=["input"], output_names=["output"])
    print(y)