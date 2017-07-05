#!/usr/bin/env python3

import argparse
import os
import numpy as np
import pdb

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
# plt.style.use('bmh')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('expDir', type = str)
    args = parser.parse_args()

    trainP = os.path.join(args.expDir, 'train.csv')
    trainData = np.loadtxt(trainP, delimiter=',').reshape(-1, 3)
    testP = os.path.join(args.expDir, 'test.csv')
    testData = np.loadtxt(testP, delimiter=',').reshape(-1, 3)

    trainI, trainLoss, trainPPL = np.split(trainData, [1, 2], axis=1)
    testI, testLoss, testPPL = np.split(testData, [1, 2], axis = 1)


    fig, ax = plt.subplots(1, 1, figsize = (10, 8))
    plt.plot(trainI, trainLoss, label = 'Train')
    plt.plot(testI, testLoss, label = 'Test')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(min(trainI), max(trainI) + 1, 1.0))
    plt.ylabel('Loss')
    plt.legend()
    loss_fname = os.path.join(args.expDir, 'loss.png')
    plt.savefig(loss_fname)
    print('Created {}'.format(loss_fname))

    fig, ax = plt.subplots(1, 1, figsize = (10, 8))
    plt.plot(trainI, trainPPL, label = 'Train')
    plt.plot(testI, testPPL, label = 'Test')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(min(trainI), max(trainI) + 1, 1.0))
    plt.ylabel('Perplexity')
    plt.legend()
    ppl_fname = os.path.join(args.expDir, 'perplexity.png')
    plt.savefig(ppl_fname)
    print('Created {}'.format(ppl_fname))


if __name__ == '__main__':
    main()