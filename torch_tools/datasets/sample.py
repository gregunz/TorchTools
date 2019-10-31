import random

from torch.utils.data import Dataset


class Sample(Dataset):
    def __init__(self, dataset: Dataset, num_samples: int = None, random_sampling=False):
        assert num_samples is None or num_samples > 0, 'need at least one sample in the dataset'

        self.dataset = dataset
        n = len(dataset)

        if num_samples is None:
            num_samples = n

        if random_sampling:
            self.indices = [random.randint(0, n - 1) for _ in range(num_samples)]
        else:
            self.indices = [i % n for i in range(num_samples)]

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
