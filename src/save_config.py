import argparse
import json
import os

PATH_PROJ = os.path.dirname(os.path.abspath(os.path.dirname('__file__')))
PATH_CONF = os.path.join(PATH_PROJ, 'configs')

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, help="Configuration JSON file name")
args = parser.parse_args()

savedir = os.path.join(PATH_CONF, args.file + '.json')

if __name__ == '__main__':
    configs = {
        'num_parties': 10,
        'staging': 1,
        'stg_shift': 2,
        'rounds': 30,
        'stg1_ep': 1,
        'stg2_ep': 10,
        'stg3_ep': 10,
        'local_bs': 256,
        'lr': 0.001,
        'momentum': 0.9,
        'event_threshold': 1,
        'unforgotten_ratio': 0.2,
        'test_every': 10,
        'seed': 42,
        'gpu': 0,
    }

    with open(savedir, 'w') as f:
        json.dump(configs, f, indent=4, separators=(',', ': '))
