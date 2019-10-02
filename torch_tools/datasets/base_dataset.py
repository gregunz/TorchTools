import random
from abc import abstractmethod
from argparse import ArgumentParser
from typing import Tuple

from torch.utils.data import Dataset, Subset

from torch_tools import utils
from .util import percentage_rdm_split


class BaseDataset(Dataset, utils.AddArgs):
    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @property
    def name(self) -> str:
        """
        Returns the name of the datasets.
        The name is usually simply defines by the name of the dataset class

        :return:
        """
        return self.__class__.__name__

    def split(self, val_percentage, seed: int = None) -> Tuple[Subset, Subset]:
        """
        Split deterministically the dataset into TRAINING and VALIDATION Subsets

        :param val_percentage:
        :param seed:
        :return:
        """
        prev_seed = random.randint(0, 1e9)
        utils.set_seed(0 if seed is None else seed)
        tng_data, val_data = percentage_rdm_split(self, val_percentage)
        utils.set_seed(prev_seed)
        return tng_data, val_data

    @staticmethod
    @abstractmethod
    def add_args(parser: ArgumentParser):
        default_val_percentage = 0.1
        parser.add_argument('--val_percentage', type=float, default=default_val_percentage,
                            help=f'percentage of data used for validation (default: {default_val_percentage})')
