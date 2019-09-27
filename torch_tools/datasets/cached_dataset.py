from torch.utils.data import Dataset
from tqdm.auto import tqdm


class CachedDataset(Dataset):
    """
    Given a dataset, creates the same dataset which caches its items.

    Note that data is not cloned/copied from the initial dataset.
    """
    def __init__(self, dataset, init_caching=False):
        self.wrapped_dataset = dataset
        self.cache = dict()
        if init_caching:
            for idx, data in enumerate(tqdm(self.wrapped_dataset)):
                self.cache[idx] = data

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]
        data = self.wrapped_dataset[index]
        self.cache[index] = data
        return data

    def __len__(self):
        return len(self.wrapped_dataset)
