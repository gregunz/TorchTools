from torch.utils.data import Dataset


class LazyTransformedDataset(Dataset):
    """
    Given a dataset, creates a dataset which applies a transform function
    to its items lazily (only when item is called).

    Note that data is not cloned/copied from the initial dataset.
    """
    def __init__(self, dataset, transform):
        self.wrapped_dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(*self.wrapped_dataset.__getitem__(index))

    def __len__(self):
        return len(self.wrapped_dataset)
