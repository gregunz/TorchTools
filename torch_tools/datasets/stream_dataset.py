from torch.utils.data.dataset import Dataset


class StreamDataset(Dataset):
    def __init__(self, dataset: Dataset):
        super().__init__()
        self.dataset = dataset
        self.n = len(dataset)

    def __getitem__(self, idx):
        return self.dataset[idx % self.n]

    def __len__(self):
        raise NotImplementedError('A stream has no length')
