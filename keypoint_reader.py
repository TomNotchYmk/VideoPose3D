import numpy as np
import sys
import argparse
from enum import Enum
np.set_printoptions(threshold=sys.maxsize)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='keypoint reader')
    parser.add_argument('-r', '--read', default=None, type=str, help="read .npz/.npy keypoint file", metavar="FILE", required=True)
    args = parser.parse_args()

    keypoint = np.load(args.read, allow_pickle=True)
#    for point in enumerate(keypoint):
    print(keypoint.shape)
    print(keypoint)