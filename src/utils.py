import copy

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

from sampling import iid, noniid


class AddGaussianNoise(object):
    def __init__(self, mean=0.5, std=0.5):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.normal(self.mean, self.std, size=tensor.size())

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class AddSaltandPepperNoise(object):
    def __init__(self, mean=0.5, std=0.5):
        self.minimum = (0 - mean) / std
        self.maximum = (1 - mean) / std

    def __call__(self, tensor):
        bernou = torch.bernoulli(torch.empty(tensor.size()).uniform_(0, 1))
        tensor[bernou == 1] = self.minimum
        bernou = torch.bernoulli(torch.empty(tensor.size()).uniform_(0, 1))
        tensor[bernou == 1] = self.maximum
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + '(min={0}, max={1})'.format(self.minimum, self.maximum)


class AddSpeckleNoise(object):
    def __init__(self, mean=0.5, std=0.5):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.normal(self.mean, self.std, size=tensor.size()) * tensor

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def get_dataset(opts, data_dir, mean=0.5, std=0.5):
    data_dir = data_dir

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean, mean, mean), (std, std, std))
    ])
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(data_dir, train=False, download=False, transform=transform)

    # Noise
    if opts['noise'] == 'gaussian':
        noise = AddGaussianNoise(mean, std)
    elif opts['noise'] == 'saltpepper':
        noise = AddSaltandPepperNoise(mean, std)
    else:
        noise = AddSpeckleNoise(mean, std)
    # Add noise
    transform_noise = transforms.Compose([
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=[-0.5, 0.5]),
        transforms.RandomCrop(20),
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((mean, mean, mean), (std, std, std)),
        noise
    ])
    noised_dataset = datasets.CIFAR10(data_dir, train=True, download=False, transform=transform_noise)

    # Get user groups either in IID or non-IID
    if opts['iid']:
        user_groups = iid(train_dataset, opts['num_users'])
    else:
        user_groups = noniid(train_dataset, opts['num_users'])
    return train_dataset, test_dataset, noised_dataset, user_groups


def concat_dataset(opts, train_dataset, noised_dataset):
    randidxs = np.random.randint(low=0, high=len(train_dataset),
                                 size=int(len(train_dataset) * (1 - opts['noise_frac'])))
    train_subset = torch.utils.data.Subset(train_dataset, randidxs)
    randidxs = np.random.randint(low=0, high=len(noised_dataset),
                                 size=int(len(noised_dataset) * opts['noise_frac']))
    noised_subset = torch.utils.data.Subset(noised_dataset, randidxs)
    concat_dataset = torch.utils.data.ConcatDataset([train_subset, noised_subset])
    return concat_dataset


def average_weights(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def plots(ylabel, savedir, **kwargs):
    colors = ['m', 'g', 'c']  # magenta, green, cyan
    matplotlib.use('Agg')

    plt.figure()
    plt.title('%s by Communicative Rounds' % ylabel)
    for i, (k, v) in enumerate(kwargs.items()):
        plt.plot(range(len(v)), v, color=colors[i], label=k)
    plt.ylabel(ylabel)
    plt.xlabel('Communication Rounds')
    plt.legend(loc='upper left')
    plt.savefig(savedir)
