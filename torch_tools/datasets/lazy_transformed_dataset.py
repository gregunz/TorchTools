from torch.utils.data import Dataset


class LazyTransformedDataset(Dataset):
    """
    Given a dataset, creates a dataset which applies a transform function
    to its items lazily (only when item is called).

    Note that data is not cloned/copied from the initial dataset.
    """
    def __init__(self, dataset: Dataset, transforms):
        self.wrapped_dataset = dataset
        self.transforms = transforms

    def __getitem__(self, index):
        return self.transforms(*self.wrapped_dataset[index])

    def __len__(self):
        return len(self.wrapped_dataset)
