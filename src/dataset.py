import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets


class idxCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root=root, train=train, transform=transform,
                         target_transform=target_transform, download=download)
        self.indices = np.arange(self.__len__())
        self.forgotten = np.zeros(len(self.indices))

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        idx = self.indices[index]
        event = self.forgotten[index]
        return image, label, idx, event


class DatasetSplit(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label, idx, event = self.dataset[self.indices[index]]
        return image, label, idx, event
