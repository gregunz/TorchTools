from torch.utils.data import Dataset


class Superset(Dataset):
    def __init__(self, dataset, num_times):
        self.dataset = dataset
        assert num_times > 0, 'cannot upsample 0 or less times'
        n = len(self.dataset)
        self.indices = [i % n for i in range(num_times * n)]

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
