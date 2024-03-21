from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Iterable, Optional

from regex import D
from torch.utils.data import DataLoader, RandomSampler, Subset, ChainDataset, StackDataset, ConcatDataset, Dataset
from gen.utils.data_defs import InputData

from gen.utils.logging_utils import log_info, log_warn
from torch import Generator
import torch
import torch.multiprocessing as mp

class Split(Enum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2


class AbstractDataset(ABC):
    """
    AbstractDataset is a class that sets up a dataset, including the dataloader.
    A typical use-case involves inheriting from AbstractDataset as well as PyTorch's Dataset class. Children must implement the get_dataset method, in this case simply returning self.
    However, this structure supports more complex datasets, including for the child to configure a dataset object defined in another library, etc. in a standardized format.
    The class allows for a unified way to configure the dataset/dataloader with subsets, shuffling, etc.
    """
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
            repeat_dataset_n_times: Optional[int] = None,
            return_tensorclass: bool = False,
            use_cuda: bool = False,
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
        self.repeat_dataset_n_times = repeat_dataset_n_times
        self.return_tensorclass = return_tensorclass
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

        # Subclasses may control these properties inside get_dataset
        self.allow_shuffle = True
        self.allow_subset = True

    @abstractmethod
    def get_dataset(self):
        pass

    def process_batch(self, batch):
        return InputData.from_dict(batch) if self.return_tensorclass else batch

    def collate_fn(self, batch):
        batch = torch.utils.data.default_collate(batch)
        batch = self.process_batch(batch)
        return batch

    def get_dataloader(self, generator: Optional[Generator] = None, pin_memory: bool = True, additional_datasets: Optional[Iterable[Dataset]] = None):
        if self.device != torch.device('cpu'):
            log_info(f"Setting start method to spawn for multi-processing. Using device: {self.device}", main_process_only=False)
            mp.set_start_method("spawn")

        orig_dataset = self.get_dataset()
        if additional_datasets is not None:
            log_info(f"Concatenating additional datasets: {additional_datasets}")
            orig_dataset = ConcatDataset([orig_dataset, *additional_datasets])

        if self.allow_subset and self.subset_size is not None:
            if self.random_subset:
                dataset = Subset(orig_dataset, list(RandomSampler(orig_dataset, num_samples=self.subset_size, generator=generator)))
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

        extra_kwargs = dict()
        if self.repeat_dataset_n_times is not None:
            dataset = ConcatDataset(*[[dataset] * self.repeat_dataset_n_times])
        else:
            extra_kwargs['shuffle'] = self.allow_shuffle and self.shuffle

        if self.drop_last:
            log_warn("Dropping last batch if it exists")

        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size, 
            collate_fn=self.collate_fn, 
            num_workers=self.num_workers, 
            pin_memory=pin_memory, 
            drop_last=self.drop_last,
            **extra_kwargs
        )
        log_info(f"Dataset size: {len(orig_dataset)}, Dataset size after subset: {len(dataset)}, Combined dataloader size (all GPUs): {len(dataloader)}")
        return dataloader