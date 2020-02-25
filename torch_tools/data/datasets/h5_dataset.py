import h5py
import torch
from torch.utils.data import Dataset


class H5Dataset(Dataset):
    def __init__(self, filepath: str, inputs_name: str = 'inputs', targets_name: str = 'targets'):
        self.filepath = filepath
        self.inputs_name = inputs_name
        self.targets_name = targets_name

    def __getitem__(self, index):
        with h5py.File(self.filepath, mode='r', libver='latest', swmr=True) as h5_file:
            if index >= len(self):
                raise IndexError

            inputs = torch.from_numpy(h5_file[self.inputs_name][index])
            targets = h5_file[self.targets_name][index]
            if sum(targets.shape) > 1:
                targets = torch.from_numpy(targets)
            else:
                targets = torch.tensor(targets)
            return inputs, targets

    def __len__(self):
        with h5py.File(self.filepath, mode='r', libver='latest', swmr=True) as h5_file:
            return h5_file[self.inputs_name].shape[0]
