import math

import numpy as np
import matplotlib.pyplot as plt

import scipy.io

sigmoid = np.vectorize(lambda i: 1 / (1 + math.exp(-i)))

relu = np.vectorize(lambda i: max(0, i))

tanh = np.vectorize(lambda i: math.tanh(i))

d_relu = np.vectorize(lambda i: 1 if i > 0 else 0)

d_sigmoid = np.vectorize(lambda i: sigmoid(i) * (1 - sigmoid(i)))

d_tanh = np.vectorize(lambda i: 1 - math.pow(math.tanh(i)))


def plot(x, y):
    # print( np.hstack((x, y)))
    blue_x = []
    blue_y = []
    red_x = []
    red_y = []
    for i in range(len(x)):
        if y.A[i, 0] == 0:
            red_x.append(x.A[i, 0])
            red_y.append(x.A[i, 1])
        else:
            blue_x.append(x.A[i, 0])
            blue_y.append(x.A[i, 1])
    plt.plot(blue_x, blue_y, 'b.', red_x, red_y, 'rs')
    plt.title('close the window to continue')
    plt.show()

def read(filename):
    result = scipy.io.loadmat(filename)
    return np.asmatrix(result['X_test'].T), \
           np.asmatrix(result['X_train'].T), \
           np.asmatrix(result['X_validation'].T), \
           np.asmatrix(result['Y_test'].T), \
           np.asmatrix(result['Y_train'].T), \
           np.asmatrix(result['Y_validation'].T)


class NNv2:
    def __init__(self, input_units, hidden_units, output_units, active_func=sigmoid, derivative_func=d_sigmoid):
        self.__hidden_units = hidden_units
        self.__output_units = output_units

        self._w1 = np.asmatrix(np.random.normal(size=(hidden_units, input_units)))
        self._b1 = np.asmatrix(np.random.normal(size=(hidden_units, 1)))

        self._w2 = np.asmatrix(np.random.normal(size=(output_units, hidden_units)))
        self._b2 = np.asmatrix(np.random.normal(size=(output_units, 1)))

        self._active_func = active_func
        self._derivative_func = derivative_func
        self._sigmoid = np.vectorize(sigmoid)

        # self._tanh = np.vectorize(math.tanh)
        # self._relu = np.vectorize(lambda x: max(0, x))

        self._x = None  # input layer

        self._z1 = None
        self._a1 = None  # hidden layer

        self._z2 = None
        self._a2 = None  # output layer

    def forward(self, x):
        assert isinstance(x, np.matrix)

        # layer0 = self._sigmoid(x)
        self._x = x

        self._z1 = self._w1 * x + self._b1
        self._a1 = self._active_func(self._z1)

        assert self._z1.shape == (self.__hidden_units, x.shape[1]), (self._z1.shape, (self.__hidden_units, x.shape[1]))
        assert self._a1.shape == (self.__hidden_units, x.shape[1]), (self._a1.shape, (self.__hidden_units, x.shape[1]))

        self._z2 = self._w2 * self._a1 + self._b2
        self._a2 = self._sigmoid(self._z2)

        assert self._z2.shape == (self.__output_units, x.shape[1]), (self._z2.shape, (self.__output_units, x.shape[1]))
        assert self._a2.shape == (self.__output_units, x.shape[1]), (self._a2.shape, (self.__output_units, x.shape[1]))

        return self._a2

    def predict(self, x):
        y = self.forward(x)
        print(y)

    def compute_cost(self, output, y, λ):
        sample_count = y.shape[1]

        total_errors = np.sum(np.power(output - y, 2) / 2) / sample_count

        regularization_cost = λ * (np.sum(np.power(self._w1, 2)) + np.sum(np.power(self._w2, 2  ))) / (2 * sample_count)

        return total_errors + regularization_cost

    def test(self, x, y, λ=1):
        output = self.forward(x)
        # classifier = np.vectorize(lambda a: 1 if a >= 0.5 else 0)
        # output1 = classifier(output)
        #
        return self.compute_cost(output, y, λ)

        # print(np.hstack((output.T, output1.T, y.T)))

    def inspect(self, title=None):
        if title is not None: print(title)
        print('\tlayer input -> hidden:')
        print('\t\t\tweight', self._w1.tolist())
        print('\t\t\tbias', self._b1.tolist())

        print('\tlayer hidden -> output:')
        print('\t\t\tweight', self._w2.tolist())
        print('\t\t\tbias', self._b2.tolist())

    def back_propagate(self, x, y, learning_rate=0.05, λ=0):
        output = self.forward(x)

        output_sigmoid_derivative = np.multiply((1 - output), output)
        output_deltas = np.multiply(output_sigmoid_derivative, output - y)

        assert output_deltas.shape == (self.__output_units, 1)

        # hidden_derivative = np.multiply(1 - self._a1, self._a1)
        hidden_derivative = np.multiply(1 - self._a1, self._a1)
        hidden_deltas_0 = np.multiply(self._w2.T, output_deltas)
        hidden_deltas = np.multiply(hidden_deltas_0, hidden_derivative)

        assert hidden_deltas.shape == (self.__hidden_units, 1), "hidden_deltas shape is (%d, %d)" % hidden_deltas.shape

        self._w2 -= learning_rate * ((output_deltas * self._a1.T) + (λ * self._w2 / len(x)))
        self._w1 -= learning_rate * ((hidden_deltas * self._x.T) + (λ * self._w1 / len(x)))

        # self._b2 = np.sum(output_deltas, axis=1)
        # self._b1 = np.sum(hidden_deltas, axis=1)

        # print(hidden_deltas)

        return np.sum(np.square(output - y)) / 2

    def train(self, x, y, batch_size=20, iterations=5000, λ=1):
        for i in range(iterations):
            total_error = 0.0
            rand_indexes = np.random.randint(0, len(x), batch_size)
            for j in range(len(rand_indexes)):
                total_error += self.back_propagate(x[:, j], y[:, j], λ)
            if i % 50 == 0:
                self.test(x, y)
                # print('total error: ', self.test(x, y))


if __name__ == "__main__":
    X_tst, X_train, X_vad, Y_tst, Y_train, Y_vad = read("./dataset.mat")

    plot(X_train, Y_train)

    active_functions = [('sigmoid', sigmoid, d_sigmoid), ('tanh', tanh, d_tanh), ('relu', relu, d_relu)]

    for hidden_num in (2, 10):
        best_score = 10000000000
        best_args = []
        best_nn =  None

        for fn in active_functions:
            for λ in (0.05, 1, 5, 10, 100):
                nn = NNv2(2, hidden_num, 1, *fn[1:])
                nn.train(X_train.T, Y_train.T, λ=λ)

                cost = nn.test(X_vad.T, Y_vad.T)
                if best_score > cost:
                    best_score = cost
                    best_args = [hidden_num, *fn[1:], fn[0], λ]
                    best_nn = nn
                    # print(best_args, best_args)

                print("\ns2 = %d, active function: %s, λ=%f" % (hidden_num, fn[0], λ))
                # nn.inspect()
                print('\tvalidate error: ', cost)

        print('\n**************************************')
        print('best λ = %f, s2 = %d, active function %s\n' % (best_args[4], best_args[0], best_args[3]))

        # nn = NNv2(2, best_args[0], 1, best_args[1], best_args[1])

        # nn.train(X_train.T, Y_train.T, λ=λ)

        best_nn.inspect()

        print('train error:', best_nn.test(X_train.T, Y_train.T, λ=λ))
        print('test error:', best_nn.test(X_tst.T, X_tst.T, λ=λ))
