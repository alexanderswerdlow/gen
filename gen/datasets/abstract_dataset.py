from abc import ABC, abstractmethod
from enum import Enum
from math import e
from typing import Any, Iterable, Optional

import torch
import torch.multiprocessing as mp
import torch.utils.data
from regex import D
from torch import Generator
from torch.utils.data import ChainDataset, ConcatDataset, DataLoader, Dataset, IterableDataset, RandomSampler, StackDataset, Subset
from torch.utils.data.sampler import Sampler

from gen.utils.data_defs import InputData
from gen.utils.logging_utils import log_info, log_warn


class Split(Enum):
    TRAIN = 0
    VALIDATION = 1
    TEST = 2

def get_even_subset(dataset, subset_size):
    idxs = list(range(len(dataset)))

    if subset_size > len(dataset):
        pass
    elif subset_size == 1:
        idxs = [len(dataset) // 2]
    else:
        step = (len(idxs) - 1) / (subset_size - 1)
        idxs = [idxs[int(round(step * i))] for i in range(subset_size)]
        
    dataset = Subset(dataset, idxs)
    return dataset

class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

    def __len__(self):
        return 1000000000

class FastDataLoader(torch.utils.data.dataloader.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class IterableSubset(IterableDataset):
    def __init__(self, dataset, indices):
        super().__init__()
        self.dataset = dataset
        self.indices = indices

    def __iter__(self):
        dataset_iter = iter(self.dataset)
        for _ in range(len(self)):
            yield next(dataset_iter)

    def __len__(self):
        return len(self.indices)

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
            split: Optional[Any] = None, 
            num_workers: int = 2, 
            batch_size: int = 2, 
            shuffle: bool = True, 
            subset_size: Optional[int] = None,
            random_subset: bool = True,
            drop_last: bool = True,
            repeat_dataset_n_times: Optional[int] = None,
            return_tensorclass: bool = False,
            use_cuda: bool = False,
            overfit_subset_size: Optional[int] = None,
            repeat_single_dataset_n_times: Optional[int] = None,
            allowed_keys: Optional[Iterable[str]] = None
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
        self.overfit_subset_size = overfit_subset_size
        self.repeat_single_dataset_n_times = repeat_single_dataset_n_times
        self.allowed_keys = allowed_keys

        # Subclasses may control these properties inside get_dataset
        self.allow_shuffle = True
        self.allow_subset = True
        self.additional_datasets = None

    @abstractmethod
    def get_dataset(self):
        pass

    def process_batch(self, batch):
        return InputData.from_dict(batch) if self.return_tensorclass else batch

    def collate_fn(self, batch):
        if self.allowed_keys is not None:
            batch = [{k:v for k,v in el.items() if k in self.allowed_keys} for el in batch]
        batch = torch.utils.data.default_collate(batch)
        batch = self.process_batch(batch)
        return batch

    def get_dataloader(
            self,
            generator: Optional[Generator] = None,
            pin_memory: bool = True,
            additional_datasets: Optional[dict[str, Dataset]] = None,
            subset_range: Optional[Any] = None,
            **kwargs
        ):
        if self.device != torch.device('cpu'):
            log_info(f"Setting start method to spawn for multi-processing. Using device: {self.device}", main_process_only=False)
            mp.set_start_method("spawn")

        if additional_datasets is not None:
            self.additional_datasets = additional_datasets

        _dataloader_cls = DataLoader
        orig_dataset = self.get_dataset()
        if self.overfit_subset_size is not None and self.subset_size is None:
            orig_dataset = get_even_subset(orig_dataset, self.overfit_subset_size)
            _dataloader_cls = FastDataLoader

        if self.repeat_single_dataset_n_times is not None:
            orig_dataset = ConcatDataset(*[[orig_dataset] * self.repeat_single_dataset_n_times])

        if self.additional_datasets is not None:
            log_info(f"Concatenating additional datasets: {self.additional_datasets}")
            for _ds_name, _ds in dict(primary=orig_dataset, **self.additional_datasets).items():
                log_info(f"Dataset {_ds_name} has size: {len(_ds)}")
            orig_dataset = ConcatDataset([orig_dataset, *self.additional_datasets.values()])

        is_iterable = isinstance(orig_dataset, IterableDataset)
        if self.allow_subset and self.subset_size is not None:
            if is_iterable:
                dataset = IterableSubset(orig_dataset, list(range(self.subset_size)))
            elif self.random_subset:
                dataset = Subset(orig_dataset, list(RandomSampler(orig_dataset, num_samples=self.subset_size, generator=generator)))
            else:
                dataset = get_even_subset(orig_dataset, self.subset_size)
        elif subset_range is not None:
            dataset = Subset(orig_dataset, subset_range)
        else: 
            dataset = orig_dataset

        extra_kwargs = dict()
        extra_kwargs.update(kwargs)
        if self.repeat_dataset_n_times is not None:
            dataset = ConcatDataset(*[[dataset] * self.repeat_dataset_n_times])
        elif is_iterable is False:
            extra_kwargs['shuffle'] = self.allow_shuffle and self.shuffle

        if self.drop_last:
            log_warn("Dropping last batch if it exists")

        dataloader = _dataloader_cls(
            dataset,
            batch_size=self.batch_size, 
            collate_fn=self.collate_fn, 
            num_workers=self.num_workers, 
            pin_memory=pin_memory, 
            drop_last=self.drop_last,
            persistent_workers=False,
            **extra_kwargs
        )
        log_info(f"Dataset size: {len(orig_dataset)}, Dataset size after subset: {len(dataset)}, Combined dataloader size (all GPUs): {len(dataloader)}")
        return dataloader