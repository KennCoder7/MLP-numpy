import pickle
from tool import *


class MLP(object):
    def __init__(self, in_dim, out_dim, hidden_dim, activation,
                 regularization='l2',
                 regularization_parameter=0.01,
                 initialization='normal',
                 name='mlp'):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.layer_nums = 0
        self.act = activation
        if out_dim == 1:
            self.output_method = 'logistic'
        else:
            self.output_method = 'softmax'
        self.regularization = regularization
        self.rgl_para = regularization_parameter
        self.initialization = initialization
        self.name = name

        self.dim = []
        self.W = {}
        self.delta_W = {}
        self.b = {}
        self.delta_b = {}
        self.z = {}
        self.a = {}
        self.e = {}  # nonsense
        self.initial()
        self.epsilon = 1e-8

        self.__train_x = None
        self.__train_y = None

        self.train_acc_log = []
        self.train_loss_log = []

    def initial(self):
        self.dim.append(self.in_dim)
        self.dim.extend(self.hidden_dim)
        self.dim.append(self.out_dim)
        self.layer_nums = len(self.dim) - 1
        for layer in range(1, self.layer_nums + 1):
            in_dim = self.dim[layer - 1]
            out_dim = self.dim[layer]
            if self.initialization == 'zeros':
                self.W[layer] = np.zeros([out_dim, in_dim])
                self.b[layer] = np.zeros([out_dim])
            elif self.initialization == 'ones':
                self.W[layer] = np.ones([out_dim, in_dim])
                self.b[layer] = np.ones([out_dim])
            elif self.initialization == 'normal':
                self.W[layer] = np.random.normal(loc=0., scale=1., size=[out_dim, in_dim])
                self.b[layer] = np.random.normal(loc=0., scale=1., size=[out_dim])
            elif self.initialization == 'xavier_Glorot_normal':
                self.W[layer] = np.random.normal(loc=0., scale=1., size=[out_dim, in_dim]) / np.sqrt(in_dim)
                self.b[layer] = np.random.normal(loc=0., scale=1., size=[out_dim]) / np.sqrt(in_dim)
            elif self.initialization == 'xavier_normal':
                std = np.sqrt(2. / (in_dim + out_dim))
                self.W[layer] = np.random.normal(loc=0., scale=std, size=[out_dim, in_dim])
                self.b[layer] = np.random.normal(loc=0., scale=std, size=[out_dim])
            elif self.initialization == 'uniform':
                a = np.sqrt(1. / in_dim)
                self.W[layer] = np.random.uniform(low=-a, high=a, size=[out_dim, in_dim])
                self.b[layer] = np.random.uniform(low=-a, high=a, size=[out_dim])
            elif self.initialization == 'xavier_uniform':
                a = np.sqrt(6. / (in_dim + out_dim))
                self.W[layer] = np.random.uniform(low=-a, high=a, size=[out_dim, in_dim])
                self.b[layer] = np.random.uniform(low=-a, high=a, size=[out_dim])
            else:
                print("initialization error!")
                exit(1)
            self.delta_W[layer] = np.zeros([out_dim, in_dim])
            self.delta_b[layer] = np.zeros([out_dim])
            self.z[layer] = np.zeros(out_dim)
            self.a[layer] = np.zeros(out_dim)
            self.e[layer] = np.zeros(out_dim)

    def print_model(self):
        print("Input dim:[{}] Hidden dim:[{}] Output dim:[{}]".format(self.in_dim, self.hidden_dim, self.out_dim))

    @staticmethod
    def af(x, activation):
        """activation_function"""
        if activation == 'sigmoid':
            return sigmoid(x)
        elif activation == 'relu':
            return relu(x)
        else:
            return x

    @staticmethod
    def daf(x, activation):
        """derivative_activation_function"""
        if activation == 'sigmoid':
            return (1 - sigmoid(x)) * sigmoid(x)
        elif activation == 'relu':
            return d_relu(x)
        else:
            return np.ones([x.shape])

    def fit(self, sample_set, target_set):
        self.__train_x = sample_set
        self.__train_y = target_set

    def forward(self, sample):
        if len(sample) != self.in_dim:
            print("len(inputs) != self.in_dim")
            exit(1)
        x = sample
        self.z[1] = np.dot(self.W[1], x) + self.b[1]
        self.a[1] = self.af(self.z[1], activation=self.act)
        for layer in range(2, self.layer_nums + 1):
            self.z[layer] = np.dot(self.W[layer], self.a[layer - 1]) + self.b[layer]
            self.a[layer] = self.af(self.z[layer], activation=self.act)
        if self.output_method == 'softmax':
            self.a[self.layer_nums] = softmax(self.z[self.layer_nums])
        else:
            self.a[self.layer_nums] = sigmoid(self.z[self.layer_nums])
        return self.a[self.layer_nums]

    def __loss(self, sample, target):
        predict_probability = self.forward(sample)
        target = np.array(target)
        if self.output_method == 'softmax':
            return np.sum(-target * np.log(predict_probability + self.epsilon))
        else:
            return np.sum(
                -target * np.log(predict_probability + self.epsilon) - (1 - target) *
                np.log(1 - predict_probability + self.epsilon))

    def backward(self, target):
        if self.output_method == 'softmax' and len(target) != self.out_dim:
            print("len(true) != self.out_dim")
            exit(1)
        target = np.array(target)
        # the output layer
        if self.output_method == 'softmax':
            for i in range(len(target)):
                if target[i] == 1:
                    self.e[self.layer_nums][i] = self.a[self.layer_nums][i] - 1
                else:
                    self.e[self.layer_nums][i] = self.a[self.layer_nums][i]
        else:
            self.e[self.layer_nums] = (self.a[self.layer_nums] - target)
        # the hidden layers
        for layer in range(self.layer_nums - 1, 0, -1):
            error = np.dot(self.W[layer + 1].transpose(1, 0), self.e[layer + 1])
            if layer != 1:
                self.e[layer] = error * self.daf(self.z[layer], activation=self.act)
            else:  # input layer
                self.e[layer] = error

    def gradient(self, sample):
        self.a[0] = np.array(sample)
        for layer in range(1, self.layer_nums + 1):
            self.delta_W[layer] = np.outer(self.e[layer], self.a[layer - 1])
            self.delta_b[layer] = self.e[layer]

    def __batch_gradient(self, sample_set, target_set):
        sample_nums = len(sample_set)
        dw = {}
        db = {}
        for layer in range(1, self.layer_nums + 1):
            dw[layer] = np.zeros([self.dim[layer], self.dim[layer - 1]])
            db[layer] = np.zeros([self.dim[layer]])
        for i in range(0, sample_nums):
            self.forward(sample_set[i])
            self.backward(target_set[i])
            self.gradient(sample_set[i])
            for layer in range(1, self.layer_nums + 1):
                dw[layer] += self.delta_W[layer]
                db[layer] += self.delta_b[layer]
                if self.regularization == 'l2':
                    dw[layer] += self.rgl_para * self.W[layer]
                    db[layer] += self.rgl_para * self.b[layer]
                elif self.regularization == 'l1':
                    dw[layer] += self.rgl_para * np.sign(self.W[layer])
                    db[layer] += self.rgl_para * np.sign(self.b[layer])
        return dw, db

    def __gradient_descent(self, lr, sample_set, target_set, vw, vb, momentum):
        """compute the gradient of every sample -> sum -> mean"""
        sample_nums = len(sample_set)
        dw, db = self.__batch_gradient(sample_set, target_set,)
        for layer in range(1, self.layer_nums + 1):
            vw[layer] = momentum * vw[layer] - lr * dw[layer] / sample_nums
            vb[layer] = momentum * vb[layer] - lr * db[layer] / sample_nums
            self.W[layer] += vw[layer]
            self.b[layer] += vb[layer]
        return vw, vb

    def train_SGD(self, lr, momentum=0.9, max_epoch=1000, batch_size=64, shuffle=True, interval=100):
        if self.__train_x is None:
            print("None data fit!")
            exit(1)
        vw = {}
        vb = {}
        for layer in range(1, self.layer_nums + 1):
            vw[layer] = np.zeros([self.dim[layer], self.dim[layer - 1]])
            vb[layer] = np.zeros([self.dim[layer]])
        batch_nums = len(self.__train_x) // batch_size
        for e in range(max_epoch):
            if shuffle and e % batch_nums == 0:
                shuffle_set(self.__train_x, self.__train_y)
            start_index = e % batch_nums * batch_size
            t_x = self.__train_x[start_index:start_index + batch_size]
            t_y = self.__train_y[start_index:start_index + batch_size]
            vw, vb = self.__gradient_descent(lr, t_x, t_y, vw, vb, momentum)
            if interval and e % interval == 0:
                train_loss = 0
                for i in range(len(t_x)):
                    train_loss += self.__loss(self.__train_x[i], self.__train_y[i])
                train_loss /= len(self.__train_x)
                train_acc = self.measure(self.__train_x, self.__train_y)
                self.train_loss_log.append(train_loss)
                self.train_acc_log.append(train_acc)
                print('Epoch[{}] Train_Loss=[{}] Train_Acc=[{}]'.format(e, train_loss, train_acc))

    def measure(self, sample_set, target_set):
        true_nums = 0
        if self.output_method == 'softmax':
            for i in range(len(sample_set)):
                pred_prob = self.forward(sample_set[i])
                if np.argmax(target_set[i]) == np.argmax(pred_prob):
                    true_nums += 1
        else:
            for i in range(len(sample_set)):
                pred_prob = self.forward(sample_set[i])
                if target_set[i] == 1 and pred_prob >= 0.5:
                    true_nums += 1
                if target_set[i] == 0 and pred_prob < 0.5:
                    true_nums += 1
        return true_nums / len(sample_set)

    def feature(self, sample_set):
        ft = np.zeros([len(sample_set), self.dim[-2]])
        for i in range(len(sample_set)):
            self.forward(sample_set[i])
            ft[i] = self.a[self.layer_nums - 1]
        return ft

    def save_training_log(self, path):
        train_log = {
            'train_loss': self.train_loss_log,
            'train_accuracy': self.train_acc_log
        }
        with open(f'{path}/{self.name}_train_log.pkl', 'wb') as f:
            pickle.dump(train_log, f, pickle.HIGHEST_PROTOCOL)

    def save_parameters(self, path):
        parameters = {
            'W': self.W,
            'b': self.b
        }
        with open(path + f'/{self.name}_parameters.pkl', 'wb') as f:
            pickle.dump(parameters, f, pickle.HIGHEST_PROTOCOL)

    def load_parameters(self, path, name):
        with open(path + f'/{name}_parameters.pkl', 'rb') as f:
            parameters = pickle.load(f)
            self.W = parameters['W']
            self.b = parameters['b']


if __name__ == '__main__':
    pass
