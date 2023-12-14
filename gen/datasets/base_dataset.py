from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from gen.configs import BaseConfig
from enum import Enum

class Split(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2

class AbstractDataset(ABC):
    def __init__(self, cfg: BaseConfig):
        self.cfg = cfg

    @abstractmethod
    def get_dataset(self, split: Enum):
        pass

    @abstractmethod
    def collate_fn(self, batch):
        pass

    def get_dataloader(self, split: Enum):
        if split != Split.TRAIN:
            raise NotImplementedError(f"Dataset split {split} not implemented.")
        
        # TODO: Fix for train/val
        dataset = self.get_dataset(split)
        return DataLoader(dataset, batch_size=self.cfg.dataset.train_batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=self.cfg.dataset.dataloader_num_workers)
