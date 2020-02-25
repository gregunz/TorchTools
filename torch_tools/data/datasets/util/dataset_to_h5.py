from itertools import tee
from pathlib import Path
from typing import Union, List, Tuple

import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

T = List[Tuple[torch.Tensor, torch.Tensor]]


def dataset_to_h5_file(dataset: Union[Dataset, T], filepath: Union[str, Path], inputs_type=None,
                       targets_type=None, inputs_name: str = 'inputs', targets_name: str = 'targets'):
    n = len(dataset)
    x, y = next(tee(iter(dataset))[1])

    assert isinstance(x, torch.Tensor), f'input should be a torch tensor, not {type(x)}'
    assert isinstance(y, torch.Tensor), f'target should be a torch tensor, not {type(x)}'

    inputs_shape = (n,) + x.size()
    targets_shape = (n,) + y.size()

    if inputs_type is None:
        inputs_type = x.numpy().dtype
    if targets_type is None:
        targets_type = y.numpy().dtype

    with h5py.File(name=filepath, mode='w', libver='latest', swmr=True) as h5_file:

        inputs = h5_file.create_dataset(inputs_name, shape=inputs_shape, dtype=inputs_type, fillvalue=0)
        targets = h5_file.create_dataset(targets_name, shape=targets_shape, dtype=targets_type, fillvalue=0)

        dloader = DataLoader(dataset, batch_size=1, num_workers=8)
        for i, (x, y) in enumerate(tqdm(dloader, desc=str(filepath))):
            inputs[i] = x
            targets[i] = y

    assert i == n - 1
