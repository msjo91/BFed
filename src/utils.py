import copy

import matplotlib
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms

from dataset import idxCIFAR10


def get_dataset(data_dir, mean=0.5, std=0.5):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean, mean, mean), (std, std, std))
    ])
    train_dataset = idxCIFAR10(data_dir, train=True, transform=transform, download=True)
    test_dataset = datasets.CIFAR10(data_dir, train=False, transform=transform, download=False)

    return train_dataset, test_dataset


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
