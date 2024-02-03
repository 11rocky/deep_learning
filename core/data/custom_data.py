import dataclasses
from dataclasses import dataclass
import torch


@dataclass
class InputData:
    input: torch.Tensor = None
    label: torch.Tensor = None
    filename: str = None

    @classmethod
    def collate_fn(cls, batch):
        data_dict = dataclasses.asdict(batch[0])
        data_keys = data_dict.keys()
        batch_data = {}
        for key in data_keys:
            if isinstance(data_dict[key], torch.Tensor):
                batch_data[key] = torch.stack([getattr(x, key) for x in batch], dim=0)
            else:
                batch_data[key] = [getattr(x, key) for x in batch]
        return cls(**batch_data)

    def to(self, *args, **kwargs):
        items = {}
        for key, value in self.__dict__.items():
            if isinstance(value, torch.Tensor) and value is not None:
                items[key] = value.to(*args, **kwargs)
        for key, value in items.items():
            setattr(self, key, value) 
        return self


@dataclass
class OutputData:
    predict: torch.Tensor = None
    aux_logits1: torch.Tensor = None
    aux_logits2: torch.Tensor = None

