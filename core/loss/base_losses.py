import torch.nn as nn
from core.data.custom_data import InputData, OutputData


class L1Loss(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(L1Loss, self).__init__()
        self.loss = nn.L1Loss(**kwargs)
    
    def forward(self, pred_data: OutputData, input_data: InputData):
        return self.loss(pred_data.predict, input_data.label)


class CrossEntropyLoss(nn.Module):
    def __init__(self, **kwargs) -> None:
        super(CrossEntropyLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(**kwargs)
    
    def forward(self, pred_data: OutputData, input_data: InputData):
        return self.loss(pred_data.predict, input_data.label)
