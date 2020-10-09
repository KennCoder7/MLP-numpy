from datasets.MNIST import MNIST
from models.MLP import MLP
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.neural_network import MLPClassifier
NAME = 'mlp-64-64-64-relu-xavier_normal'


def train_mlp(name):
    mlp = MLP(
        in_dim=784,
        out_dim=10,
        hidden_dim=[64, 64, 64],
        activation='relu',
        initialization='xavier_normal',
        regularization='l2',
        name=name
    )
    dt = MNIST()
    dt.normalization()
    mlp.fit(dt.train_x, dt.train_y)
    mlp.train_SGD(
        lr=0.05,
        momentum=0.9,
        max_epoch=501,
        batch_size=128,
        interval=10
    )
    print('Test_Acc=[{}]'.format(mlp.measure(dt.test_x, dt.test_y)))
    mlp.save_parameters('log/Mnist/model_parameters')
    # mlp.save_training_log('log/Mnist/training_log')


def plot_training_log(name):
    with open(f'log/Mnist/training_log/{name}_train_log.pkl', 'rb') as f:
        train_log = pickle.load(f)
    training_loss = train_log['train_loss']
    training_acc = train_log['train_accuracy']
    plt.subplot(121)
    plt.plot(np.arange(len(training_loss)), training_loss)
    plt.title('Loss')
    plt.subplot(122)
    plt.plot(np.arange(len(training_acc)), training_acc)
    plt.title('Accuracy')
    plt.suptitle(name)
    plt.show()


def reload_mlp(name):
    mlp = MLP(
        in_dim=784,
        out_dim=10,
        hidden_dim=[64, 64, 64],
        activation='relu'
    )
    mlp.load_parameters('log/Mnist/model_parameters', name)
    dt = MNIST()
    dt.normalization()
    print('Test_Acc=[{}]'.format(mlp.measure(dt.test_x, dt.test_y)))
    ft = mlp.feature(dt.test_x)
    print('Feature shape', ft.shape)
    np.save(f'log/Mnist/feature/{name}_ft.npy', ft)


def sklearn_mlp():
    mlp = MLPClassifier(
        hidden_layer_sizes=(100),
        activation='relu',
        solver='sgd',
        batch_size=64,
        learning_rate_init=0.05,
        max_iter=2001
    )
    dataset = MNIST(dimension=1)
    mlp.fit(dataset.train_x, dataset.train_labels)
    train_acc = mlp.score(dataset.train_x, dataset.train_labels)
    test_acc = mlp.score(dataset.test_x, dataset.test_labels)
    print(f'Sklearn_MLP: Train acc=[{train_acc}] Test acc=[{test_acc}]')
    return


if __name__ == '__main__':
    # train_mlp(NAME)
    # plot_training_log(NAME)
    reload_mlp(NAME)
    # sklearn_mlp()
