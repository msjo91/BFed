import argparse
import os

import numpy as np

PATH_PROJ = os.path.dirname(os.path.abspath(os.path.dirname('__file__')))
PATH_RES = os.path.join(PATH_PROJ, 'results')
PATH_PERF = os.path.join(PATH_RES, 'performances')

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', type=str, help="NPY file name")
args = parser.parse_args()

accdir = os.path.join(PATH_PERF, args.file + '_test_acc')
lsdir = os.path.join(PATH_PERF, args.file + '_test_ls')

if __name__ == '__main__':
    acc = np.load(accdir + '.npy').reshape(-1)
    loss = np.load(lsdir + '.npy').reshape(-1)
    print('Test Accuracy: ', acc)
    print('Test Loss: ', loss)

    np.savetxt(accdir + '.csv', acc, delimiter=',')
    # np.savetxt(lsdir + '.csv', loss, delimiter=',')
