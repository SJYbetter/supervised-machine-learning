# *coding:UTF-8*
import random

import matplotlib.pyplot as plt
import numpy as np
import scipy.io


def read_matlab2(filename):
    result = scipy.io.loadmat(filename)
    return result['X_trn'], result['Y_trn'], result['X_tst'], result['Y_tst']


def _nextBatch(samples, checks, batchSize=10):
    index = {}
    while len(index) < batchSize:
        rowNo = random.randint(0, samples.shape[0] - 1)
        index[rowNo] = rowNo
    index = list(index.keys())
    print('rand index:', index)
    return samples[index, :], checks[index, :]


def _shuffle(samples, checks):
    columnCount = samples.shape[1]
    newModel = samples.hstack(checks)
    np.random.shuffle(newModel)

    return newModel[:, 0: columnCount], newModel[:, newModel:]


def closedForm(X, Y):
    return ((X.T * X).I) * (X.T) * Y


# def SGD(X, Y, loopLimit, learnRate):
#     m, n = np.shape(X)
#     weights = np.ones([n, 1])
#     for i in range(loopLimit):
#         np.random.shuffle(X)
#         for j in X:
#             h = samples * weights - Y
#         weights = weights - learnRate * (X.T * h)
#     # print(weights)
#     return weights

def _default_debug_function(iterationId, weights):
    pass


def debug_print(iterationId, weights):
    print(iterationId, weights[:, 0])


def BatchGD(samples, checks, loopLimit, learnRate, reportfn=_default_debug_function):
    """ 批量梯度下降算法，一次处理所有训练样本 """
    m, n = samples.shape
    weights = np.ones([n, 1])
    for i in range(loopLimit):
        error = samples * weights - checks
        weights -= learnRate * (samples.T * error) / (2 * m)
        reportfn(i, weights)
    return weights


def SGD(samples, checks, loopLimit, learnRate, reportfn=_default_debug_function):
    """ 随机梯度下降法， 一次只随机训练一个样本 """
    m, n = np.shape(samples)
    weights = np.ones([n, 1])
    for i in range(loopLimit):
        rowIndex = random.randint(0, m - 1)
        one_sample, one_check = samples[rowIndex, :], checks[rowIndex, :]
        weights -= learnRate * one_sample.T * (one_sample * weights - one_check) / (2 * m)
        reportfn(i, weights)
    return weights


def MiniBatchSGD(samples, checks, loopLimit, learnRate, batchSize, reportfn=_default_debug_function):
    """  最小批随机梯度下降法， 一次随机选择一批样本训练 """
    m, n = samples.shape
    weights = np.ones([n, 1])
    for iterationId in range(loopLimit):
        randIndex = list(range(0, m, batchSize))
        np.random.shuffle(randIndex)
        for index in randIndex:
            batch_samples = samples[index: index + batchSize, :]
            batch_checks = checks[index: index + batchSize, :]
            #
            # if batch_samples.shape[0] > batchSize:
            #     print(index, batch_samples.shape)
            #     raise AssertionError("batch size mismatch %d <-> %d", batch_samples.shape[0], batchSize)
            #
            error = batch_samples * weights - batch_checks
            weights -= learnRate * (batch_samples.T * error) / (2 * m)

            # print('MiniBatchSGD index: ', index)
        reportfn(iterationId, weights)
        # if error[0:0] == -np.inf or error[0:0] is np.nan:
        #    break
        # ee = learnRate * (batch_samples.T * error) / m
        # print(weights, ee, error)
    return weights


def MiniBatchSGD_v2(samples, checks, loopLimit, learnRate, batchSize, reportfn=_default_debug_function):
    """  最小批随机梯度下降法， 一次随机选择一批样本训练。
        与MiniBatchSGD区别：
            MiniBatchSGD_v2，每次迭代，仅选择一部分数据，进行训练，
            MiniBatchSGD 每次迭代，分批遍历所有数据
    """
    m, n = samples.shape
    thetas = np.ones([n, 1])

    for iterationId in range(loopLimit):
        rand_indexes = np.random.randint(0, m, batchSize)

        batch_samples = samples[rand_indexes, :]
        batch_checks = checks[rand_indexes, :]

        error = batch_samples * thetas - batch_checks
        thetas -= learnRate * (batch_samples.T * error) / (2 * m)

        reportfn(iterationId, thetas)
    return thetas


def plot(X, Y, weights):
    """ 绘图，画出所有的点和回归直线 """
    ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['bottom'].set_position(('data', 0))
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_position(('data', 0))

    plt.plot(X.T.A[1], Y.T.A[0], '.', label="samples")

    x = X.T.A[1]
    y = (X.T[1] * weights[1] + weights[0]).A[0]
    x.sort()
    y.sort()
    plt.plot(x, y, label='w: %s' % weights)
    plt.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':
    X_trn, Y_trn, X_tst, Y_tst = read_matlab2("dataset/dataset1.mat")
    X_trn = np.mat(np.insert(X_trn, 0, 1, axis=1))
    Y_trn = np.mat(Y_trn)

    # 批量梯度下降
    # weights = BatchGD(X_trn, Y_trn, 3000, 0.01)

    # 随机梯度下降
    # weights = SGD(X_trn, Y_trn, 3000, 0.01)

    # 随机小批量梯度下降
    weights = MiniBatchSGD(X_trn, Y_trn, 10000, 0.01, 7, debug_print)

    # 绘制结果
    plot(X_trn, Y_trn, weights[:, 0])
