import math

import matplotlib.pyplot as plt

__figureId = 0


def draw_samples(samples, checks, thetas, title):
    global __figureId
    m, n = samples.shape
    subplot_base = math.ceil(n / 2) * 100 + 20

    __figureId += 1
    fig = plt.figure(__figureId)
    fig.suptitle(title + " sample data")
    plt.subplots_adjust(hspace=0.8)
    for i in range(n):
        plt.subplot(subplot_base + i + 1)
        plt.plot(samples[:, i], checks[:, 0], '.')
        plt.title('col_%d' % i)
        if i == 0: continue
        x = samples[:, i].A


def plot_result_error(result, plot_id):
    plt.subplots_adjust(hspace=0.8)

    plot_id[2] += 1
    plt.subplot(*plot_id)
    plt.title(result.title + 'train error')
    plt.plot(range(len(result.train_errors)), result.train_errors)

    plot_id[2] += 1
    plt.subplot(*plot_id)
    plt.title(result.title + 'test error')
    plt.plot(range(len(result.test_errors)), result.test_errors)


def plot_contest(contest):
    global __figureId

    sgd_result = contest.get_sgd_result()
    keys = tuple(sgd_result.keys())
    thetas = sgd_result[keys[-1]] if len(keys) > 0 else None

    draw_samples(contest.X_trn, contest.Y_trn, thetas, contest.title)

    __figureId += 1
    fig = plt.figure(__figureId)
    fig.suptitle(contest.title)

    plot_id = [len(sgd_result), 2, 0]
    for key, result in sgd_result.items():
        plot_result_error(result, plot_id)


def show():
    plt.show()
