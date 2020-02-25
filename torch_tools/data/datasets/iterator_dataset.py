from collections import Iterator
from itertools import tee
from typing import List, Union

from torch.utils.data import Dataset


class IteratorDataset(Dataset):
    def __init__(self, iterator: Union[List, Iterator], length: int = None, repeat_when_empty=True):
        if length is None:  # iterator might have a length (e.g. is a list)
            try:
                length = len(iterator)
            except TypeError:
                pass
        self.__iterator_original, self.__iterator = tee(iterator)
        self.__length = length
        self.__repeat_when_empty = repeat_when_empty

    def __getitem__(self, idx):
        try:
            return next(self.__iterator)
        except StopIteration as e:
            if not self.__repeat_when_empty:
                raise e
            self.__iterator_original, self.__iterator = tee(self.__iterator_original)
            return self.__getitem__(idx)

    def __len__(self):
        # if self.__length is None:
        #     raise ValueError('This iterator has unknown length. Define it at instantiation.')
        return self.__length

    def __iter__(self):
        return tee(self.__iterator_original)[1]
