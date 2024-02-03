import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_chnls, out_chnls, kernel_size, stride=1, padding=0, has_bias=False,
                 act=None, *args, **kwargs) -> None:
        super().__init__()
        self._conv = nn.Conv2d(in_chnls, out_chnls, kernel_size, stride, padding,
                               has_bias=has_bias, *args, **kwargs)
        self._bn = nn.BatchNorm2d(out_chnls)
        self._act = act
    
    def forward(self, x):
        y = self._conv(x)
        y = self._bn(y)
        if self._act is not None:
            y = self._act(y)
        return y


class ResBlock(nn.Module):
    def __init__(self, in_chnls, out_chnls, stride=1, downsample=None, elt_func=torch.add) -> None:
        super(ResBlock, self).__init__()
        self._conv1 = nn.Conv2d(in_chnls, out_chnls,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self._bn1 = nn.BatchNorm2d(out_chnls)
        self._conv2 = nn.Conv2d(out_chnls, out_chnls,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self._bn2 = nn.BatchNorm2d(out_chnls)
        self._downsample = downsample
        self._elt_func = elt_func
    
    def forward(self, x):
        identity = x
        if self._downsample is not None:
            identity = self._downsample(x)
        out = F.relu(self._bn1(self._conv1(x)))
        out = self._bn2(self._conv2(out))
        return F.relu(self._elt_func(out, identity))

