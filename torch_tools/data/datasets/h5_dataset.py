import h5py
import torch
from torch.utils.data import Dataset


class H5Dataset(Dataset):
    def __init__(self, filepath: str, inputs_name: str = 'inputs', targets_name: str = 'targets'):
        h5_file = h5py.File(filepath, 'r', libver='latest', swmr=True)
        self.inputs = h5_file[inputs_name]
        self.targets = h5_file[targets_name]

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError

        inputs = torch.tensor(self.inputs[index])
        targets = torch.tensor(self.targets[index])
        return inputs, targets

    def __len__(self):
        return self.inputs.shape[0]
