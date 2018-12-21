import os
import sys

import numpy as np
import sklearn.preprocessing as preprocessing

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from ds5220_2 import dataset, _draw


def matrixBuild2(source, repeatLimit=5, scale=1):
    resultArray = []
    for one_row_in_source in source:
        x = one_row_in_source[0]
        # new_row = [1, x]
        # for i in range(2, repeatLimit + 1):
        #    new_row.append(pow(x, i))
        new_row = [pow(x, i) * scale for i in range(repeatLimit + 1)]
        resultArray.append(new_row)
    # print(resultArray)
    # np.random.shuffle([12, 3, 1212])
    return np.mat(resultArray)


def closed_form(X, Y):
    k2 = np.dot(X.T, X)
    k22 = np.dot(np.linalg.inv(k2), X.T)
    return np.dot(k22, Y)


def _default_debug_function(iterationId, weights):
    pass


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


def summary_square_errors(samples, checks, weights):
    errors = samples * weights - checks
    return ((errors.T * errors) / errors.shape[0] / 2)[0, 0]


def format_number_array(ndarr):
    return ', '.join([str(i) for i in ndarr])


class ResultData:
    def __init__(self, title, rate=100):
        self.title = title
        self.thetaList = []
        self.train_errors = []
        self.test_errors = []
        self.last_theta = None
        self.last_train_error = None
        self.last_test_error = None
        self.__rate = rate

    def __iter__(self):
        return iter(self.thetaList)

    def append_theta(self, iter_id, theta):
        if (iter_id % self.__rate) == 0:
            self.thetaList.append(theta.copy())

    def append_errors(self, train_error, test_error):
        self.train_errors.append(train_error)
        self.test_errors.append(test_error)


class Contest:
    def __init__(self, title, X_trn, Y_trn, X_tst, Y_tst):
        self.title = title
        self.X_trn = X_trn
        self.Y_trn = Y_trn
        self.X_tst = X_tst
        self.Y_tst = Y_tst
        self._results = {}
        self._closed_form_result = None
        self._learningRate = None
        self._iterationLimit = None

    def get_sgd_result(self):
        return self._results

    def run_different_algorithms_v2(self, learningRate=0.01, iterationLimit=3000,
                                    batchList=[3, 30, 60]):
        self._learningRate = learningRate
        self._iterationLimit = iterationLimit

        # 计算closed form结果
        cfrslt = ResultData("closed form")
        cfrslt.last_theta = closed_form(self.X_trn, self.Y_trn)
        cfrslt.last_train_error = summary_square_errors(self.X_trn, self.Y_trn, cfrslt.last_theta)
        cfrslt.last_test_error = summary_square_errors(self.X_tst, self.Y_tst, cfrslt.last_theta)
        self._closed_form_result = cfrslt

        for oneBatch in batchList:
            title = "batch %d iteration %d learningRate %f " % (oneBatch, iterationLimit, learningRate)
            result = ResultData(title)
            result.last_theta = MiniBatchSGD_v2(self.X_trn, self.Y_trn, iterationLimit, learningRate, oneBatch,
                                                reportfn=result.append_theta)
            result.last_train_error = summary_square_errors(self.X_trn, self.Y_trn, result.last_theta)
            result.last_test_error = summary_square_errors(self.X_tst, self.Y_tst, result.last_theta)
            self._results[oneBatch] = result

        # 用于显示error下降曲线
        for key, result in self._results.items():
            for theta in result:
                train_error = summary_square_errors(self.X_trn, self.Y_trn, theta)
                test_error = summary_square_errors(self.X_tst, self.X_tst, theta)
                result.append_errors(train_error, test_error)

    def show_result(self):
        print("%s:\t\tlearningRate: %f\titeration: %d" % (self.title, self._learningRate, self._iterationLimit))

        print("\tclosed form:\t train error: %f\t test error: %f\ttheta: %s" % (
            self._closed_form_result.last_train_error,
            self._closed_form_result.last_test_error,
            format_number_array(self._closed_form_result.last_theta.A[:, 0])))

        for b, result in self._results.items():
            print("\t     sgd_%d:\t train error: %f\t test error: %f\ttheta: %s" %
                  (b, result.last_train_error, result.last_test_error,
                   format_number_array(result.last_theta[:, 0])))

        _draw.plot_contest(self)

    def run(self, learning_rate=0.01, iteration_limit=3000, batch_list=[3, 30, 60]):
        self.run_different_algorithms_v2(learning_rate, iteration_limit, batch_list)
        self.show_result()


def main3(X_trn, Y_trn, X_tst, Y_tst):
    scaler = preprocessing.MinMaxScaler((-1, 1), True).fit(X_trn)

    x_trn_scaled = scaler.transform(X_trn)
    x_tst_scaled = scaler.transform(X_tst)
    y_trn_scaled = scaler.transform(Y_trn)
    y_tst_scaled = scaler.transform(Y_tst)

    Contest("X_pow_2", matrixBuild2(X_trn, 2), Y_trn, matrixBuild2(X_tst, 2), Y_tst).run()
    Contest("X_pow_3", matrixBuild2(X_trn, 3), Y_trn, matrixBuild2(X_tst, 3), Y_tst).run(
        iteration_limit=100000, learning_rate=0.01)

    Contest("X_pow_5", matrixBuild2(x_trn_scaled, 5), y_trn_scaled, matrixBuild2(x_tst_scaled, 5), y_tst_scaled).run(
        iteration_limit=100000, learning_rate=0.3)

    _draw.show()


if __name__ == "__main__":
    X_Trn, Y_Trn, X_Tst, Y_Tst = dataset.read_matrix(dataset.FILE0)
    main3(X_Trn, Y_Trn, X_Tst, Y_Tst)
