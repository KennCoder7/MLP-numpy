import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    shift_x = x - np.max(x)  # 防止输入增大时输出为nan
    exp_x = np.exp(shift_x)
    return exp_x / np.sum(exp_x)


def relu(x):
    return np.maximum(0, x)


def d_relu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def shuffle_set(sample_set, target_set):
    index = np.arange(len(sample_set))
    np.random.shuffle(index)
    return sample_set[index], target_set[index]
