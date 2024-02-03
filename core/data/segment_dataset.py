from torch.utils import data
from .custom_data import InputData


class SegmentDataset(data.Dataset):
    def __init__(self, opt) -> None:
        super(SegmentDataset, self).__init__()
        self._data = [1, 2, 3]
    
    def __len__(self):
        return len(self._data)

    def __getitem__(self, index) -> InputData:
        data = InputData()
        return data
