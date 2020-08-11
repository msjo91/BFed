import argparse
import copy
import json
import os
import random
import time
from datetime import timedelta

import numpy as np
import torch
from torch.utils.data import DataLoader

from learn import train, test
from local import Party
from models import CifarCNN
from utils import get_dataset

# Paths
PATH_PROJ = os.path.dirname(os.path.abspath(os.path.dirname('__file__')))
PATH_DATA = os.path.join(PATH_PROJ, 'data')
PATH_RES = os.path.join(PATH_PROJ, 'results')
PATH_PERF = os.path.join(PATH_RES, 'performances')
PATH_PLOT = os.path.join(PATH_RES, 'plots')
PATH_CONF = os.path.join(PATH_PROJ, 'configs')

# Argument
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, help="Option JSON file name")
args = parser.parse_args()

# Configurations
with open(os.path.join(PATH_CONF, args.file + '.json'), 'r') as f:
    config = json.load(f)

# Reproducibilty
os.environ['PYTHONHASHSEED'] = str(config['seed'])
random.seed(config['seed'])
np.random.seed(config['seed'])
torch.manual_seed(config['seed'])

# Device
device = torch.device('cuda:{}'.format(config['gpu']) if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # Begin measuring runtime
    start_time = time.time()

    # Load dataset
    train_dataset, test_dataset = get_dataset(PATH_DATA, mean=0.5, std=0.5)
    testloader = DataLoader(test_dataset, batch_size=int(len(test_dataset) / 10), shuffle=False)

    # Build parties
    num_items = int(len(train_dataset) / config['num_parties'])
    indices = copy.deepcopy(train_dataset.indices)
    parties = []

    for i in range(config['num_parties']):
        allocated_idxs = np.random.choice(indices, num_items, replace=False)
        p = Party(name=i, config=config, indices=allocated_idxs)
        parties.append(p)
        indices = list(set(indices) - set(p.indices))

    # Build model
    global_model = CifarCNN()
    global_model.to(device)
    print(global_model)

    # Train
    global_model, losses = train(global_model, config, train_dataset, parties, testloader, test_every=config['test_every'])

    # Test
    test_acc, test_ls = test(global_model, config, testloader)
    print(f'\n Test Results after {config["rounds"]} global rounds of training:')
    print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))
    print("|---- Test Loss: {:.4f}".format(test_ls))

    save_dir = os.path.join(PATH_PERF, '{}_test_acc.npy'.format(args.file))
    np.save(save_dir, test_acc)
    save_dir = os.path.join(PATH_PERF, '{}_test_ls.npy'.format(args.file))
    np.save(save_dir, test_ls)

    # Print execution time
    print('\nRuntime: ', timedelta(seconds=time.time() - start_time))
