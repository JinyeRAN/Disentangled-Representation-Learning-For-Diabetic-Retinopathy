import torch
import torch.nn as nn
from solo.data.temp_dali_fix import TempDALIGenericIterator
from typing import Callable, List, Optional, Union


class PretrainWrapper(TempDALIGenericIterator):
    def __init__(
        self, cfg,
        model_batch_size: int, model_rank: int, model_device: torch.device,
        dataset_size: int, conversion_map: List[int] = None, *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cfg = cfg
        self.model_batch_size = model_batch_size
        self.model_rank = model_rank
        self.model_device = model_device
        self.dataset_size = dataset_size
        self.conversion_map = None

    def __next__(self):
        batch = super().__next__()[0]
        *all_X, targets, filename = (batch[v] for v in self.output_map)
        targets = targets.squeeze(-1).long()
        filename = filename.squeeze(-1).long()
        all_X = [x for x in all_X]
        return [all_X, targets, filename]

    def __len__(self):
        if self.cfg.data.drop_last:
            return self.dataset_size // self.model_batch_size
        else:
            return self.dataset_size // self.model_batch_size + 1


class Wrapper(TempDALIGenericIterator):
    def __init__(self, dataset_size: int, model_bacth_size:int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_size = dataset_size
        self.model_bacth_size = model_bacth_size

    def __next__(self):
        batch = super().__next__()[0]
        x, target, filename = batch["x"], batch["label"], batch["filename"]
        target = target.squeeze(-1).long()
        filename = filename.squeeze(-1).long()
        x = x.detach().clone()
        target = target.detach().clone()
        filename = filename.detach().clone()
        return x, target, filename

    def __len__(self):
        return self.dataset_size // self.model_bacth_size + 1