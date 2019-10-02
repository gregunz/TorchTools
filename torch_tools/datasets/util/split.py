from typing import Tuple

from torch.utils.data import Dataset, Subset, random_split


def percentage_rdm_split(dataset: Dataset, percentage: float) -> Tuple[Subset, Subset]:
    """
    Split a dataset into two random Subsets given a percentage.

    :param dataset:
    :param percentage:
    :return:
    """
    assert 0 < percentage < 1
    val_length = int(round(percentage * len(dataset)))
    tng_dataset, val_dataset = random_split(dataset, [len(dataset) - val_length, val_length])
    return tng_dataset, val_dataset