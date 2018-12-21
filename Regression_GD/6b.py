# *coding:UTF-8*
import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from ds5220_2 import dataset, _6a, _draw


def dump_matrix(samples, checks):
    """ 废弃，调试用途 """
    for i in range(samples.shape[0]):
        print(",".join([str(n) for n in samples.A[i, :]]))


def summary_square_errors(samples, checks, weights):
    errors = samples * weights - checks
    return ((errors.T * errors) / errors.shape[0] / 2)[0, 0]


def format_number_array(ndarr):
    return ', '.join([str(i) for i in ndarr])


class ResultData:
    def __init__(self, title):
        self.title = title
        self.thetaList = []
        self.train_errors = []
        self.test_errors = []
        self.last_theta = None
        self.last_train_error = None
        self.last_test_error = None

    def __iter__(self):
        return iter(self.thetaList)

    def append_theta(self, iter_id, theta):
        self.thetaList.append(theta.copy())

    def append_errors(self, train_error, test_error):
        self.train_errors.append(train_error)
        self.test_errors.append(test_error)


class Contest:
    _figureId = 0

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

    def run_different_algorithms_v2(self, learning_rate=0.01, iteration_limit=3000,
                                    batch_list=[3, 30, 60]):
        self._learningRate = learning_rate
        self._iterationLimit = iteration_limit

        cfrslt = ResultData("closed form")
        cfrslt.last_theta = _6a.closedForm(self.X_trn, self.Y_trn)
        cfrslt.last_train_error = summary_square_errors(self.X_trn, self.Y_trn, cfrslt.last_theta)
        cfrslt.last_test_error = summary_square_errors(self.X_tst, self.Y_tst, cfrslt.last_theta)
        self._closed_form_result = cfrslt

        for oneBatch in batch_list:
            title = "batch %d iteration %d learningRate %f " % (oneBatch, iteration_limit, learning_rate)
            result = ResultData(title)
            result.last_theta = _6a.MiniBatchSGD_v2(self.X_trn, self.Y_trn, iteration_limit, learning_rate, oneBatch,
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

        # 调用绘图功能，显示 train data 和 test data 的cost/error下降曲线
        _draw.plot_contest(self)

    def run(self, learning_rate=0.01, iteration_limit=3000, batch_list=[3, 30, 60]):
        self.run_different_algorithms_v2(learning_rate, iteration_limit, batch_list)
        self.show_result()


def main3(X_trn, Y_trn, X_tst, Y_tst):
    x_trn_scaled, y_trn_scaled, x_tst_scaled, y_tst_scaled = dataset.min_max_scale(X_trn, X_trn, Y_trn, X_tst, Y_tst)

    Contest("X_pow_2", dataset.build_matrix_by_pow(X_Trn, 2), Y_Trn, dataset.build_matrix_by_pow(X_Tst, 2), Y_Tst).run()
    Contest("X_pow_3", dataset.build_matrix_by_pow(X_Trn, 3), Y_Trn, dataset.build_matrix_by_pow(X_Tst, 3), Y_Tst).run()

    Contest("X_pow_5", dataset.build_matrix_by_pow(x_trn_scaled, 5), y_trn_scaled, dataset.build_matrix_by_pow(x_tst_scaled, 5),
            y_tst_scaled).run(batch_list=[1, 3, 7, 9, 10, 20, 30])

    plt.show()


if __name__ == "__main__":
    X_Trn, Y_Trn, X_Tst, Y_Tst = dataset.read_matrix(dataset.FILE0)

    main3(X_Trn, Y_Trn, X_Tst, Y_Tst)
