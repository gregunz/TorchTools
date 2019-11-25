import torch
from torch.utils.data import Dataset


def compute_mean_std(dataset: Dataset, dim=0, n_passes=1):
    dataset_size = dataset[0][0].size()
    channels = dataset_size[dim]
    dims = [i for i in range(len(dataset_size)) if i != dim]

    running_mean = torch.zeros(channels)
    running_std = torch.zeros(channels)
    n = torch.zeros(1)

    for _ in range(n_passes):
        for inputs, _ in dataset:
            running_mean += inputs.mean(dim=dims)
            running_std += inputs.std(dim=dims)
            n += 1

    return running_mean / n, running_std / n
