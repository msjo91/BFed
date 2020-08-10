import argparse
import json
import os
import time
from datetime import timedelta

import numpy as np
import torch
from tensorboardX import SummaryWriter

from models import CifarCNN
from run import federated, solo
from utils import get_dataset

# Paths
PATH_PROJ = os.path.dirname(os.path.abspath(os.path.dirname('__file__')))
PATH_DATA = os.path.join(PATH_PROJ, 'data')
PATH_LOG = os.path.join(PATH_PROJ, 'logs')
PATH_RES = os.path.join(PATH_PROJ, 'results')
PATH_PERF = os.path.join(PATH_RES, 'performances')
PATH_PLOT = os.path.join(PATH_RES, 'plots')
PATH_OPT = os.path.join(PATH_PROJ, 'options')

# Argument
parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, help="Option JSON file name")
args = parser.parse_args()

# Options
with open(os.path.join(PATH_OPT, args.file + '.json'), 'r') as f:
    opts = json.load(f)

# TensorboardX logger
logger = SummaryWriter(PATH_LOG)

# Device
device = torch.device('cuda:{}'.format(opts['gpu']) if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    # Begin measuring runtime
    start_time = time.time()

    # Load dataset and client groups
    train_dataset, test_dataset, noised_dataset, user_groups = get_dataset(opts, PATH_DATA)

    # Build model
    global_model = CifarCNN()
    global_model.to(device)
    print(global_model)

    if opts['federated'] == 0:
        te_acc, te_ls = solo(global_model, opts, train_dataset, test_dataset, noised_dataset)
    elif opts['federated'] == 1:
        tr_acc, tr_ls, te_acc, te_ls = federated(global_model, opts, train_dataset, test_dataset,
                                                 noised_dataset, user_groups, logger)

    save_dir = os.path.join(PATH_PERF, '{}_test_acc.npy'.format(args.file))
    np.save(save_dir, te_acc)
    save_dir = os.path.join(PATH_PERF, '{}_test_ls.npy'.format(args.file))
    np.save(save_dir, te_ls)

    # Print execution time
    print('\nRuntime: ', timedelta(seconds=time.time() - start_time))
