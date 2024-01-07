from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

from torch.utils.data import DataLoader, RandomSampler, Subset


class Split(Enum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2


class AbstractDataset(ABC):
    def __init__(
            self,
            *,
            cfg: Optional[Any] = None, 
            split: Split, 
            num_workers: int = 2, 
            batch_size: int = 2, 
            shuffle: bool = True, 
            random_subset: Optional[int] = None, 
        ):
        from gen.configs import BaseConfig
        self.cfg: BaseConfig = cfg
        self.split: Split = split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_subset = random_subset

        # Subclasses may control these properties inside get_dataset
        self.allow_shuffle = True
        self.allow_random_subset = True

    @abstractmethod
    def get_dataset(self):
        pass

    @abstractmethod
    def collate_fn(self, batch):
        pass

    def get_dataloader(self):
        orig_dataset = self.get_dataset()
        if self.allow_random_subset and self.random_subset is not None:
            dataset = Subset(orig_dataset, list(RandomSampler(orig_dataset, num_samples=self.random_subset)))
        else: 
            dataset = orig_dataset
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.allow_shuffle and self.shuffle, collate_fn=self.collate_fn, num_workers=self.num_workers, pin_memory=True)
        print(f"Original dataset size: {len(orig_dataset)}, Final dataset size: {len(dataset)}, Dataloader size: {len(dataloader)}")
        return dataloader