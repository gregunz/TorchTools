from torch.utils.data import Dataset


class Superset(Dataset):
    def __init__(self, dataset: Dataset, num_samples: int):
        assert num_samples > 0, 'need at least one sample in the dataset'

        self.dataset = dataset
        n = len(dataset)
        self.indices = [i % n for i in range(num_samples)]

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
