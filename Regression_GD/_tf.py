import os
import sys

import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from ds5220_2 import dataset


def do_train(samples, checks):
    m, n = samples.shape
    # Model linear regression y = Wx + b
    x = tf.placeholder(tf.float64, samples.shape, name="samples")
    y = tf.placeholder(tf.float64, checks.shape, name="checks")

    # x = tf.Variable(samples.A, name="samples")
    # y = tf.Variable(checks.A, name="checks")

    theta = tf.Variable(tf.zeros([n, 1], tf.float64), name="theta")
    b = tf.Variable(tf.zeros([1], tf.float64), name="bias")

    hypothesis = tf.matmul(x, theta) + b

    # Cost function sum((y_-y)**2)
    cost = tf.reduce_sum(tf.square(hypothesis - y))

    # Training using Gradient Descent to minimize cost
    train_step = tf.train.GradientDescentOptimizer(0.0000001).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(3000):
            # Train
            feed = {x: samples.A, y: checks.A}
            sess.run(train_step, feed_dict=feed)
            # print("After %d iteration:" % i)
        print("W: ", sess.run(theta, feed_dict=feed))
        print("b: ", sess.run(b, feed_dict=feed))
        # # Suggested by @jihobak
        print("cost:", sess.run(cost, feed_dict=feed))


if __name__ == "__main__":
    X_Trn, Y_Trn, X_Tst, Y_Tst = dataset.read_matrix(dataset.FILE0)

    x_trn_scaled, y_trn_scaled, x_tst_scaled, y_tst_scaled = dataset.min_max_scale(X_Trn, X_Trn, Y_Trn, X_Tst, Y_Tst)

    do_train(np.mat(dataset.build_matrix_by_pow(x_trn_scaled)), np.mat(y_trn_scaled))
