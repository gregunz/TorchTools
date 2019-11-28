import random

from torch.utils.data import Dataset


class Sample(Dataset):
    def __init__(self, dataset: Dataset, num_samples: int = None, random_sampling=False):
        assert num_samples is None or num_samples > 0, 'need at least one sample in the dataset'

        self.dataset = dataset
        n = len(dataset)

        if num_samples is None:
            num_samples = n

        self.all_indices = list(range(num_samples))

        if random_sampling:
            random.shuffle(self.all_indices)

        self.indices = [i % n for i in self.all_indices[:num_samples]]

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

# class Sample(Dataset):
#     def __init__(self, dataset: Dataset, num_samples: int = None, random_sampling=False):
#         assert num_samples is None or num_samples > 0, 'need at least one sample in the dataset'
#
#         self.random_sampling = random_sampling
#         self.dataset = dataset
#
#         n = len(dataset)
#
#         if num_samples is None:
#             num_samples = n
#
#         self.all_indices = list(range(n))
#
#         if random_sampling:
#             random.shuffle(self.all_indices)
#             self.n_idx = 0
#
#         self.indices = [i % n for i in self.all_indices[:num_samples]]
#
#     def __getitem__(self, idx):
#         if self.random_sampling:
#             self.n_idx += 1
#             if self.n_idx % len(self) == 0:
#                 self.n_idx = 0
#                 random.shuffle(self.all_indices)
#                 self.indices = [i % len(self.dataset) for i in self.all_indices[:num_samples]]
#
#         return self.dataset[self.indices[idx]]
#
#     def __len__(self):
#         return len(self.indices)
