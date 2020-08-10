import argparse
import json
import os

PATH_PROJ = os.path.dirname(os.path.abspath(os.path.dirname('__file__')))
PATH_OPT = os.path.join(PATH_PROJ, 'options')

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, help="Configuration JSON file name")
args = parser.parse_args()

savedir = os.path.join(PATH_OPT, args.file + '.json')

if __name__ == '__main__':
    options = {
        'epochs': 30,
        'num_users': 10,
        'frac': 1,
        'local_ep': 10,
        'local_bs': 50,
        'optimizer': 'sgd',
        'lr': 0.001,
        'momentum': 0.9,
        'seed': 42,
        'noise': 'gaussian',
        'noise_frac': 0.2,
        'federated': 1,
        'iid': 1,
        'clean': 2,
        'dist': 'mahalanobis',
        'gpu': 0,
    }

    with open(savedir, 'w') as f:
        json.dump(options, f)
