# FileName:ANN_utils.py
# coding = utf-8
# Created by Hzq
import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings
import datetime
warnings.filterwarnings('ignore')


class MlpNn:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.depth = 0
        self.layer_n = list()
        self.layer_a = list()
        self.w = dict()
        self.b = dict()
        self.z = dict()
        self.a = dict()
        self.dz = dict()
        self.dw = dict()
        self.db = dict()
        self.vw = dict()
        self.vb = dict()
        self.sw = dict()
        self.sb = dict()
        self.vw_c = dict()
        self.vb_c = dict()
        self.cost = 0
        self.m = 0
        self.epoch_num = 0
        self.optimizer = ''

    def add_layer(self, neuron_num, activation='sigmoid'):
        self.layer_n.append(neuron_num)
        self.layer_a.append(activation)

    def add_dynamic_layer(self, neuron_num, activation='sigmoid'):
        pass

    def forward_propagation(self, x):
        self.a['0'] = x.reshape(-1, 1)
        for i in range(0, self.depth):
            f_obj = getattr(sys.modules[__name__], self.layer_a[i])
            self.z[str(i+1)] = np.dot(self.w[str(i+1)], self.a[str(i)]) + self.b[str(i + 1)]
            self.a[str(i+1)] = f_obj(self.z[str(i+1)])
        return self.a[str(self.depth)]

    def back_propagation(self, p, y, optimizer, learn_rate, learn_decay, loss_type):
        d_loss = getattr(sys.modules[__name__], 'd_' + loss_type)
        d_activation = getattr(sys.modules[__name__], 'd_' + self.layer_a[self.depth - 1])
        self.dz[str(self.depth)] = d_loss(p, y) * d_activation(self.a[str(self.depth)])
        learn_decay = 1. / (1. + learn_decay * self.epoch_num)
        for i in range(self.depth, 0, -1):
            d_activation = getattr(sys.modules[__name__], 'd_' + self.layer_a[i-2])
            # compute derivation
            self.dw[str(i)] = np.dot(self.dz[str(i)], self.a[str(i-1)].T)
            self.db[str(i)] = np.sum(self.dz[str(i)], axis=1, keepdims=True)
            self.dz[str(i-1)] = np.dot(self.w[str(i)].T, self.dz[str(i)]) * d_activation(self.a[str(i-1)])
            # update parameters
            if optimizer == 'momentum':
                self.vw[str(i)] = 0.9 * self.vw[str(i)] + 0.1 * self.dw[str(i)]
                self.vb[str(i)] = 0.9 * self.vb[str(i)] + 0.1 * self.db[str(i)]
                self.w[str(i)] -= learn_decay * learn_rate * self.vw[str(i)]
                self.b[str(i)] -= learn_decay * learn_rate * self.vb[str(i)]
            elif optimizer == 'gd':
                self.w[str(i)] -= learn_decay * learn_rate * self.dw[str(i)]
                self.b[str(i)] -= learn_decay * learn_rate * self.db[str(i)]
            elif optimizer == 'adam':
                self.vw[str(i)] = 0.9 * self.vw[str(i)] + 0.1 * self.dw[str(i)]
                self.vb[str(i)] = 0.9 * self.vb[str(i)] + 0.1 * self.db[str(i)]
                vw_c = self.vw[str(i)] / (1 - 0.9 ** self.epoch_num)
                vb_c = self.vb[str(i)] / (1 - 0.9 ** self.epoch_num)
                self.sw[str(i)] += (1 - 0.999) * (self.dw[str(i)] ** 2)
                self.sb[str(i)] += (1 - 0.999) * (self.db[str(i)] ** 2)
                sw_c = self.sw[str(i)] / (1 - 0.999 ** self.epoch_num)
                sb_c = self.sb[str(i)] / (1 - 0.999 ** self.epoch_num)
                self.w[str(i)] -= learn_rate * (vw_c / (np.sqrt(sw_c) + 1e-8))
                self.b[str(i)] -= learn_rate * (vb_c / (np.sqrt(sb_c) + 1e-8))
            else:
                raise Exception('Optimizer Type Error!')

    def compile(self, optimizer='momentum'):
        self.depth = len(self.layer_n)
        n_last = self.input_shape
        # initialization
        for i in range(0, self.depth):
            n = np.int16(np.array(self.layer_n[i]))
            if self.layer_a[i] == 'leaky_relu' or self.layer_a[i] == 'relu' or self.layer_a[i] == 'linear':
                self.w[str(i + 1)] = np.random.randn(n, n_last) * np.sqrt(2. / n)
            elif self.layer_a[i] == 'tanh':
                self.w[str(i + 1)] = np.random.randn(n, n_last) * np.sqrt(1. / n)
            elif self.layer_a[i] == 'sigmoid':
                self.w[str(i + 1)] = np.random.randn(n, n_last)
            else:
                raise Exception('Activation Type Error!')
            self.b[str(i + 1)] = np.zeros((n, 1))
            n_last = np.int16(np.array(self.layer_n[i]))
        self.initialize_optimizer(optimizer)
        self.optimizer = optimizer

    def fit(self, train_x, train_y, epoch,
            loss_type='square_loss', learn_rate=0.001, learn_decay=0.0002, verbose=None):
        cost_record = []
        div = epoch / 10

        self.m = train_x.shape[0]
        start_time = datetime.datetime.now()

        for eps in range(epoch):
            self.cost = 0
            self.epoch_num += 1
            for i in range(self.m):
                # forward_propagation
                x = train_x[i]
                p = self.forward_propagation(x)
                # compute_loss
                y = np.reshape(train_y[i], (-1, 1))
                loss = self.computate_loss(p, y, loss_type)
                self.cost += loss
                # back_propagation
                self.back_propagation(p, y, self.optimizer, learn_rate, learn_decay, loss_type)
            if loss_type == 'square_loss':
                self.cost = self.cost / self.m
                cost_record.append(self.cost)
            if verbose and (eps+1) % div == 0:
                current_time = datetime.datetime.now()
                eta = (current_time - start_time) * ((epoch - eps) / eps)
                print('Epoch=', eps+1, '  Cost=', np.sum(self.cost), '  ETA:', eta)
        if verbose >= 2:
            cost_record = np.array(cost_record).reshape(-1)
            plt.plot(cost_record)
            plt.xlabel('Epochs')
            plt.ylabel('Cost')
            plt.show()

    def computate_loss(self, p, y, loss_type):
        loss_fun = getattr(sys.modules[__name__], loss_type)
        return loss_fun(p, y)

    def predict(self, x):
        return np.reshape(self.forward_propagation(x), (1, -1))

    def show_predict2D(self, data_x, label_y, x_mul=1., y_mul=1., x_offset=0, y_offset=0):
        y = list()
        for i in range(data_x.shape[0]):
            y.append(self.forward_propagation(data_x[i]))
        y = np.array(y).reshape(-1)

        plt.plot(data_x * x_mul + x_offset, y * y_mul + y_offset, label='Predict')
        plt.plot(data_x * x_mul + x_offset, label_y * y_mul + y_offset, '--', label='Expect')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.show()

    def show_predict3D(self, data_x, label_y, x_mul=1., y_mul=1., x_offset=0, y_offset=0):
        from mpl_toolkits.mplot3d import Axes3D
        y = np.zeros(data_x.shape[0])
        for i in range(data_x.shape[0]):
            y[i] = self.forward_propagation(data_x[i])
        label_y = label_y.reshape(-1, 1)
        y = y.reshape(-1, 1)
        y, _, _ = normalization(y)
        fig = plt.figure()
        ax = Axes3D(fig)
        data_x = data_x * x_mul + x_offset
        y = y * y_mul + y_offset
        ax.scatter(data_x[:, 0], data_x[:, 1], y, marker='o', label='Predict')
        ax.scatter(data_x[:, 0], data_x[:, 1], label_y, marker='^', label='Expect')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('F value')
        plt.legend()
        plt.show()

    def initialize_optimizer(self, optimizer):
        if optimizer == 'momentum':
            for l in range(0, self.depth):
                self.vw[str(l+1)] = np.zeros(self.w[str(l+1)].shape)
                self.vb[str(l+1)] = np.zeros(self.b[str(l+1)].shape)
        elif optimizer == 'adam':
            for l in range(0, self.depth):
                self.vw[str(l + 1)] = np.zeros(self.w[str(l + 1)].shape)
                self.vb[str(l + 1)] = np.zeros(self.b[str(l + 1)].shape)
                self.sw[str(l + 1)] = np.zeros(self.w[str(l + 1)].shape)
                self.sb[str(l + 1)] = np.zeros(self.b[str(l + 1)].shape)

    def summary(self):
        print('Model summary : Layers = '+str(self.depth))
        for i, _ in enumerate(self.layer_n):
            print('Layer'+str(i), ', Neurons= '+str(self.layer_n[i]), ', Activation= '+self.layer_a[i])


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


def d_sigmoid(v):
    return v * (1 - v)


def tanh(x):
    return np.tanh(x)


def d_tanh(v):
    return 1 - np.square(v)


def relu(x):
    return (x + np.abs(x)) / 2.


def d_relu(v):
    return (v >= 0) * 1


def leaky_relu(x, leak=0.05):
    t1 = 1 + leak
    t2 = 1 - leak
    return (t1 * x + t2 * np.abs(x)) / 2.


def d_leaky_relu(v, leak=0.05):
    t1 = (v >= 0) * 1
    t2 = (v < 0) * leak
    return t1 + t2


def linear(x):
    return x


def d_linear(_):
    return 1.


def square_loss(p, y):
    return np.square(p-y) / 2.


def d_square_loss(p, y):
    return p-y


def normalization(x, offset=None, mul=None):
    ave = np.mean(x)
    if offset:
        x = x - offset
    else:
        x = x - ave
    std = np.std(x)
    if mul:
        x = x / mul
    else:
        x = x / std
    return x, ave, std
