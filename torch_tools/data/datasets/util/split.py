import random
from itertools import accumulate
from typing import Tuple

import torch
from torch.utils.data import Dataset, Subset

from torch_tools import utils


def split(dataset: Dataset, percentage: float, seed: int = None) -> Tuple[Subset, Subset]:
    """
    Split a dataset into two random Subsets given a percentage.

    :param dataset:
    :param percentage:
    :param seed:
    :return:
    """
    assert 0 < percentage < 1
    prev_seed = random.randint(0, 1e9)
    # this forces the split to always be the same with a fixed seed
    utils.set_seed(0 if seed is None else seed)
    val_length = int(round(percentage * len(dataset)))
    lengths = [len(dataset) - val_length, val_length]

    indices = torch.randperm(sum(lengths)).tolist()
    tng_data, val_data = [Subset(dataset, indices[offset - length:offset])
                          for offset, length in zip(accumulate(lengths), lengths)]
    # we don't want the code running after to use the fixed seed.
    utils.set_seed(prev_seed)
    return tng_data, val_data
