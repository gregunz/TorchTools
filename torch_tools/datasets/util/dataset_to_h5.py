from pathlib import Path
from typing import Union

import h5py
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


def dataset_to_h5_file(dataset: Dataset, filepath: Union[str, Path], inputs_type=None, targets_type=None,
                       inputs_name: str = 'inputs', targets_name: str = 'targets'):
    n = len(dataset)
    x, y = dataset[0]

    assert isinstance(x, torch.Tensor), 'input should be a torch tensor'
    assert isinstance(y, torch.Tensor), 'target should be a torch tensor'

    inputs_shape = (n,) + x.size()
    targets_shape = (n,) + y.size()

    if inputs_type is None:
        inputs_type = x.numpy().dtype
    if targets_type is None:
        targets_type = y.numpy().dtype

    with h5py.File(name=filepath, mode='w', libver='latest', swmr=True) as h5_file:

        inputs = h5_file.create_dataset(inputs_name, shape=inputs_shape, dtype=inputs_type, fillvalue=0)
        targets = h5_file.create_dataset(targets_name, shape=targets_shape, dtype=targets_type, fillvalue=0)

        for i, (x, y) in enumerate(tqdm(dataset)):
            inputs[i] = x
            targets[i] = y
