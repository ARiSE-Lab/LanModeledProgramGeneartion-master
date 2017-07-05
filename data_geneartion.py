import numpy as np
import operator
from collections import OrderedDict
from operator import itemgetter

def get_vocabulary(path):
    vocabulary = []

def generate_model_data(all_f, work_dir, ratio = 99):
    with open(all_f, 'r') as all_data,\
        open(work_dir+"train.txt", 'w')as train_f,\
        open(work_dir+"valid.txt", 'w')as valid_f,\
        open(work_dir+"test.txt", 'w') as test_f:
        data = all_data.readlines()
        total =  len(data)
        train_size = total * ratio // 100
        test_size = total - train_size
        print(" total: ", total, ' train_size: ', train_size, " test size: ", test_size, " data: ", data[0])
        for i in range(train_size):
            train_f.write(data[i])
        for i in range (train_size, total):
        # train_f.write(data[:train_size])
            test_f.write(data[i])
            valid_f.write(data[i])

def main():
    project_name = "maven"
    work_dir = "./soft_data/"
    generate_model_data('./soft_data/parsed_methods.txt', work_dir, ratio=90)

if __name__ == "__main__":
    main()