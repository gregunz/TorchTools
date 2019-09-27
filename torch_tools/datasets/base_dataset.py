from abc import abstractmethod, ABC

from torch.utils.data import random_split, Dataset


class BaseDataset(Dataset, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def tng_dataset(self):
        raise NotImplementedError

    @abstractmethod
    def tst_dataset(self):
        raise NotImplementedError

    @property
    def name(self):
        """
        Returns the name of the datasets.
        The name is usually simply defines by the name of the dataset class

        :return:
        """
        return self.__class__.__name__

    def test_split(self):
        """
        Split the datasets into TRAINING and TESTING BaseDataset.

        Note: Testing datasets should not be part of training/tuning of the models, i.e. it is not
        a validation datasets.

        :return:
        """
        return self.tng_dataset(), self.tst_dataset()

    def validation_split(self, val_percentage):
        """
        Split the dataset into TRAINING and VALIDATION torchvision.datasets.Subset(s)

        Note: Unlike the testing, it does NOT need to be fixed/defined (or does it ?) and
        hence, it might change over runs if the seed is not set.

        :param val_percentage:
        :return:
        """
        assert 0 < val_percentage < 1
        val_length = int(round(val_percentage * len(self)))
        return random_split(self, [len(self) - val_length, val_length])

    def split(self, val_percentage):
        """
        Split the dataset into TRAINING, VALIDATION and TESTING torchvision.datasets.Subset(s)

        :param val_percentage:
        :return:
        """
        full_tng_data, tst_data = self.test_split()
        tng_data, val_data = full_tng_data.validation_split(val_percentage)
        return tng_data, val_data, tst_data
