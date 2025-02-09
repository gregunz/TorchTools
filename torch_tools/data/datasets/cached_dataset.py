from torch.multiprocessing import Manager

from torch.utils.data import Dataset
from tqdm.auto import tqdm


class CachedDataset(Dataset):
    """
    Given a dataset, creates the same dataset which caches its items.

    Note that data is not cloned/copied from the initial dataset.
    """

    def __init__(self, dataset: Dataset, init_caching=False):
        self.dataset = dataset
        if init_caching:
            self.cache = dict()
        else:
            self.cache = Manager().dict()
        if init_caching:
            for idx, data in enumerate(tqdm(self.dataset)):
                self.cache[idx] = data

    def __getitem__(self, index):
        if index not in self.cache:
            data = self.dataset[index]
            self.cache[index] = data
        return self.cache[index]

    def __len__(self):
        return len(self.dataset)
