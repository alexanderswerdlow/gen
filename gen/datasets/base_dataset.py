from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from enum import Enum

class Split(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2

class AbstractDataset(ABC):
    def __init__(self, cfg, num_workers: int, batch_size: int, shuffle: bool, **kwargs):
        from gen.configs import BaseConfig
        self.cfg: BaseConfig = cfg
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle

    @abstractmethod
    def get_dataset(self):
        pass

    @abstractmethod
    def collate_fn(self, batch):
        pass

    def get_dataloader(self):
        # TODO: Fix for train/val
        dataset = self.get_dataset()
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle, collate_fn=self.collate_fn, num_workers=self.num_workers, pin_memory=True)
