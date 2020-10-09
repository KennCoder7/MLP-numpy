import numpy as np
import struct
import os
PATH = r'datasets/mnist_data'


class MNIST(object):
    def __init__(self, shuffle=True, dimension=1):
        self.train_x = None
        self.train_y = None
        self.train_labels = None

        self.test_x = None
        self.test_y = None
        self.test_labels = None

        self._shuffle = shuffle
        self.dim = dimension

        self.__load_mnist_train(PATH)
        self.__load_mnist_test(PATH)
        self.__one_hot()
        self.__dimension()
        if self._shuffle:
            self.shuffle()

    def __load_mnist_train(self, path, kind='train'):
        labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
        images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)
        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        self.train_x = images
        self.train_labels = labels

    def __load_mnist_test(self, path, kind='t10k'):
        labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
        images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)
        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        self.test_x = images
        self.test_labels = labels

    def __one_hot(self):
        trn = np.zeros([len(self.train_labels), 10])
        te = np.zeros([len(self.test_labels), 10])
        for i, x in enumerate(self.train_labels):
            trn[i, x] = 1
        for i, x in enumerate(self.test_labels):
            te[i, x] = 1
        self.train_y = trn
        self.test_y = te

    def normalization(self):
        self.train_x = self.train_x / 255.
        self.test_x = self.test_x / 255.

    def __dimension(self):
        if self.dim == 1:
            pass
        elif self.dim == 3:
            self.train_x = np.reshape(self.train_x, [len(self.train_x), 1, 28, 28])
            self.test_x = np.reshape(self.test_x, [len(self.test_x), 1, 28, 28])
        else:
            print('Dimension Error!')
            exit(1)

    def shuffle(self):
        index = np.arange(len(self.train_x))
        np.random.seed(7)
        np.random.shuffle(index)
        self.train_x = self.train_x[index]
        self.train_y = self.train_y[index]
        self.train_labels = self.train_labels[index]
