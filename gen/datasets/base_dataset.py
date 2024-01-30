from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional

from torch.utils.data import DataLoader, RandomSampler, Subset

from gen.utils.logging_utils import log_info, log_warn
from torch import Generator

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
            subset_size: Optional[int] = None,
            random_subset: bool = True,
            drop_last: bool = True,
        ):
        from gen.configs import BaseConfig
        self.cfg: BaseConfig = cfg
        self.split: Split = split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.subset_size = subset_size
        self.random_subset = random_subset # Either get random indices or a determinstic, evenly spaced subset
        self.drop_last = drop_last

        # Subclasses may control these properties inside get_dataset
        self.allow_shuffle = True
        self.allow_subset = True

    @abstractmethod
    def get_dataset(self):
        pass

    @abstractmethod
    def collate_fn(self, batch):
        pass

    def get_dataloader(self, g: Optional[Generator] = None):
        orig_dataset = self.get_dataset()
        if self.allow_subset and self.subset_size is not None:
            if self.random_subset:

                dataset = Subset(orig_dataset, list(RandomSampler(orig_dataset, num_samples=self.subset_size, generator=g)))
            else:
                idxs = list(range(len(orig_dataset)))

                if self.subset_size > len(orig_dataset):
                    pass
                elif self.subset_size == 1:
                    idxs = [len(orig_dataset) // 2]
                else:
                    step = (len(idxs) - 1) / (self.subset_size - 1)
                    idxs = [idxs[int(round(step * i))] for i in range(self.subset_size)]
                    
                dataset = Subset(orig_dataset, idxs)
        else: 
            dataset = orig_dataset

        if self.drop_last:
            log_warn("Dropping last batch if it exists")

        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=self.allow_shuffle and self.shuffle, collate_fn=self.collate_fn, num_workers=self.num_workers, pin_memory=True, drop_last=self.drop_last)
        log_info(f"Dataset size: {len(orig_dataset)}, Dataset size after subset: {len(dataset)}, Combined dataloader size (all GPUs): {len(dataloader)}")
        return dataloader