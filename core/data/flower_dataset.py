import os
import torch
from torch.utils import data
from PIL import Image
import torchvision.transforms as T
from .custom_data import InputData


class FlowerDataset(data.Dataset):
    def __init__(self, idx, desc, opt, train=True) -> None:
        super(FlowerDataset, self).__init__()
        self.file_list = []
        path, cls = desc[0], desc[1]
        files = os.listdir(path)
        for f in files:
            if f.endswith(".jpg") or f.endswith(".png"):
                self.file_list.append(os.path.join(path, f))
        self.label = [0] * opt.num_classes
        self.label[cls] = 1
        if train:
            self.trans = T.Compose([
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(), 
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            self.trans = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(), 
                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index) -> InputData:
        img = Image.open(self.file_list[index])
        data = InputData(
            input=self.trans(img),
            label=torch.tensor(self.label, dtype=torch.float32)
        )
        return data
