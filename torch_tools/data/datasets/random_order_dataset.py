import random

from torch.utils.data import Dataset


class RandomOrderDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.indices = list(range(len(dataset)))
        random.shuffle(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
