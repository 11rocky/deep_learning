import dataclasses
from dataclasses import dataclass
import torch
import numpy as np


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
        data_dict = dataclasses.asdict(self)
        for key, value in data_dict.items():
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                setattr(self, key, value.to(*args, **kwargs))
            elif isinstance(value, np.ndarray):
                setattr(self, key, torch.from_numpy(value).to(*args, **kwargs))
        return self


@dataclass
class OutputData:
    predict: torch.Tensor = None
    aux_logits1: torch.Tensor = None
    aux_logits2: torch.Tensor = None

