import os

import numpy as np
import scipy.io
import sklearn.preprocessing as preprocessing

curdir = os.path.dirname(os.path.realpath(__file__))

FILE0 = os.path.join(curdir, 'dataset', 'dataset1.mat')  # 'E:\\tmp\\hw02_data_DS5220\\HW1_Data\\dataset1.mat'
FILE1 = os.path.join(curdir, 'dataset', 'dataset2.mat')  # 'E:\\tmp\\hw02_data_DS5220\\HW1_Data\\dataset2.mat'


def read_matrix(filename):
    result = scipy.io.loadmat(filename)

    return result['X_trn'], result['Y_trn'], result['X_tst'], result['Y_tst']


def build_matrix_by_pow(source, repeat_limit=5):
    results = []
    for one_row_in_source in source:
        x = one_row_in_source[0]
        # new_row = [1, x]
        # for i in range(2, repeatLimit + 1):
        #    new_row.append(pow(x, i))
        new_row = [pow(x, i) for i in range(repeat_limit + 1)]
        results.append(new_row)
    return np.mat(results)


def min_max_scale(data_to_fit, *data_to_transform):
    scaler = preprocessing.MinMaxScaler((-1, 1), True)
    scaler.fit(data_to_fit)
    results = []
    for orig in data_to_transform:
        results.append(scaler.transform(orig))
    return results
