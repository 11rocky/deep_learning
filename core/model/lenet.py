import torch
import torch.nn as nn
import torch.nn.functional as F
from core.data.custom_data import InputData, OutputData


class LeNet(nn.Module):
    def __init__(self, opt) -> None:
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(opt.image_chnl, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.linear1 = nn.Linear(32 * 5 * 5, 120)
        self.linear2 = nn.Linear(120, 84)
        self.linear3 = nn.Linear(84, 10)
    
    def forward(self, data: InputData) -> OutputData:
        x = data.input
        y = F.max_pool2d(F.relu(self.conv1(x)), 2, 2)
        y = F.max_pool2d(F.relu(self.conv2(y)), 2, 2)
        y = y.view(-1, 32 * 5 * 5)
        y = F.relu(self.linear1(y))
        y = F.relu(self.linear2(y))
        y = self.linear3(y)
        return OutputData(y)


if __name__ == "__main__":
    net = LeNet()
    x = torch.rand((1, 3, 32, 32))
    y = net(x)
    print(y)

