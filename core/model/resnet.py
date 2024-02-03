import torch
import torch.nn as nn
import torch.nn.functional as F
from core.data.custom_data import InputData, OutputData


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_chnls, out_chnls, stride=1, downsample=None) -> None:
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_chnls, out_chnls,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_chnls)
        self.conv2 = nn.Conv2d(out_chnls, out_chnls,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_chnls)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


class ResNet(nn.Module):
    def __init__(self, opt) -> None:
        super(ResNet, self).__init__()
        block = BasicBlock
        self.include_top = getattr(opt, "include_top", True)
        self.in_chnl = 64
        self.conv1 = nn.Conv2d(3, self.in_chnl, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_chnl)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, opt.block_num[0])
        self.layer2 = self._make_layer(block, 128, opt.block_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, opt.block_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, opt.block_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.linear = nn.Linear(512 * block.expansion, opt.num_classes)
    
    def _make_layer(self, block, chnl, block_num ,stride=1):
        downsample = None
        if stride != 1 or self.in_chnl != chnl * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_chnl, chnl * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(chnl * block.expansion)
            )
        layers = []
        layers.append(block(self.in_chnl, chnl, downsample=downsample, stride=stride))
        self.in_chnl = chnl * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_chnl, chnl))
        return nn.Sequential(*layers)

    def forward(self, data: InputData) -> OutputData:
        x = data.input
        x = self.maxpool(F.relu(self.bn1(self.conv1(x)), True))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.include_top:
            x = self.linear(torch.flatten(self.avgpool(x), 1))
        return OutputData(x)


def resnet34(num_classes, include_top=True):
    return ResNet(BasicBlock, [2, 4, 6, 3], num_classes, include_top)


if __name__ == '__main__':
    net = ResNet(BasicBlock, [2, 4, 6, 3], 10)
    x = torch.rand((1, 3, 224, 224))
    net.eval()
    y = net(x)
    torch.onnx.export(net, x, "resnet.onnx", opset_version=11,
                      input_names=['input'], output_names=['output'])
    print(y)
