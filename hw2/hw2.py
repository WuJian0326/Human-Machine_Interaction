import numpy as np
import math
import os
import matplotlib.pyplot as plt
from mnist import MNIST


def init_weight(prev_d, cur_d):
    w = np.random.uniform(-1.0 * math.sqrt(6.0 / (prev_d + cur_d)),
                          math.sqrt(6.0 / (prev_d + cur_d)),
                          (prev_d, cur_d))
    b = np.random.randn(1, cur_d)

    return w, b


def softmax(x):
    max_per_row = np.reshape(np.max(x, axis=1), (x.shape[0], 1))
    exp_scores = np.exp(x - max_per_row)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs


def ReLU(x):
    f_x = np.maximum(0, x)
    d_x = f_x.copy()
    d_x[d_x > 0] = 1
    return f_x, d_x


def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))
    return sig, sig * (1 - sig)


def tanh(x):
    tan = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return tan, 1 - np.square(tan)


def cross_entropy(f_w3, y):
    i = range(0, f_w3.shape[0])
    L_i = -np.log(f_w3[i, y.astype(int)[i]])
    loss = 1 / L_i.shape[0] * np.sum(L_i)
    return loss


class MyNetwork:
    def __init__(self, in_d, h1_d, h2_d, out_d, lr, act_f, norm=True):
        self.in_d = in_d
        self.h1_d = h1_d
        self.h2_d = h2_d
        self.out_d = out_d
        self.lr = lr
        self.act_f = act_f
        self.norm = norm

        # Adam parameter
        self.adam_belta1 = 0.9
        self.adam_belta2 = 0.999
        self.adam_epsilon = 1e-8
        self.adam_alpha = lr

        self.train_loss_list = []
        self.test_acc_list = []
        self.test_error_list = []

        self.train_loss_file = "loss.png"
        self.test_acc_file = "test_accuracy.png"
        self.test_error_file = "test_error.png"

        # initialize layer's parameter
        np.random.seed(5)
        self.W1, self.b1 = init_weight(self.in_d, self.h1_d)
        self.W2, self.b2 = init_weight(self.h1_d, self.h2_d)
        self.W3, self.b3 = init_weight(self.h2_d, self.out_d)

    def load_data(self, train_image, train_label, test_image, test_label):
        # Process train data
        self.X_train, self.y_train = self.process_data(train_image, train_label)
        self.train_label = train_label

        # Process test data
        self.X_test, self.y_test = self.process_data(test_image, test_label)
        self.test_label = test_label

    def process_data(self, X, y):
        # X = X.transpose()
        X = X / np.max(X)
        if self.norm:
            X = (X - np.mean(X)) / np.std(X)

        one_hot = np.zeros((y.size, self.out_d))
        one_hot[np.arange(y.size), y.astype(int)] = 1

        return X, one_hot

    def load_weight(self, w1, w2, w3):
        self.W1, self.W2, self.W3 = w1, w2, w3

    def save_weight(self, w1, w2, w3):
        if not os.path.exists("weight"):
            os.mkdir("weight")
        np.save(f"weight/w1_{self.in_d}_{self.h1_d}", w1)
        np.save(f"weight/w2_{self.h1_d}_{self.h2_d}", w2)
        np.save(f"weight/w3_{self.h2_d}_{self.out_d}", w3)

    def activation(self, x):
        if self.act_f == "relu":
            return ReLU(x)
        if self.act_f == "sigmoid":
            return sigmoid(x)
        if self.act_f == "tanh":
            return tanh(x)

    def forward(self, x, dropout=0.2):
        # print(x.shape)
        z1 = x @ self.W1 + self.b1
        a1, d_z1 = self.activation(z1)

        if 0 < dropout < 1:
            mask_h1 = np.random.binomial(size=a1.shape[1], n=1, p=dropout)/dropout
        else:
            mask_h1 = 1
        a1 = a1 * mask_h1

        z2 = a1 @ self.W2 + self.b2
        a2, d_z2 = self.activation(z2)

        if 0 < dropout < 1:
            mask_h2 = np.random.binomial(size=a2.shape[1], n=1, p=dropout)/dropout
        else:
            mask_h2 = 1
        a2 = a2 * mask_h2

        z3 = a2 @ self.W3 + self.b3
        a3 = softmax(z3)

        return a1, a2, a3, d_z1, d_z2, mask_h1, mask_h2

    def backward(self, a1, a2, a3, d_z1, d_z2, x, y, m_h1, m_h2):
        d_z3 = a3 - y
        d_a3 = d_z3 @ self.W3.T

        size = x.shape[0]

        d_z2 = d_a3 * d_z2 * m_h2
        d_a2 = d_z2 @ self.W2.T

        d_z1 = d_a2 * d_z1 * m_h1

        grad_z1 = (x.T @ d_z1) / size
        grad_z2 = (a1.T @ d_z2) / size
        grad_z3 = (a2.T @ d_z3) / size

        grad_b1 = np.ones((1, d_z1.shape[0])) @ d_z1
        grad_b2 = np.ones((1, d_z2.shape[0])) @ d_z2
        grad_b3 = np.ones((1, d_z3.shape[0])) @ d_z3

        return grad_z1, grad_z2, grad_z3, grad_b1, grad_b2, grad_b3

    def calculate_error(self, a3, y):
        return cross_entropy(a3, y)

    def calculate_test_acc(self, x, y):
        a1, a2, a3, d_z1, d_z2, m_h1, m_h2 = self.forward(x, dropout=1)
        pred = (a3 == a3.max(axis=1, keepdims=1)).astype(float)
        return np.sum(np.equal(pred, y).all(axis=1)) / x.shape[0]

    def update(self, grad_w1, grad_w2, grad_w3):
        self.W1 = self.W1 - self.lr * grad_w1
        self.W2 = self.W2 - self.lr * grad_w2
        self.W3 = self.W3 - self.lr * grad_w3

    def adam_update(self, w, grad_w, m, v):
        m = self.adam_belta1 * m + (1. - self.adam_belta1) * grad_w
        v = self.adam_belta2 * v + (1. - self.adam_belta2) * (np.square(grad_w))
        w = w - self.adam_alpha * m / (np.sqrt(v) + self.adam_epsilon)
        return w, m, v

    def plot_loss(self, x, y):
        plt.figure(1)
        plt.plot(range(x), y, 'g-', label='train loss')
        plt.xticks(range(x))
        plt.legend()
        plt.title("My NN Loss")
        plt.xlabel("epoch")
        plt.ylabel("Error")

        if not os.path.exists("result"):
            os.mkdir("result")
        plt.savefig(f"result/{self.train_loss_file}")
        plt.show()

    def plot_acc(self, x, y):
        plt.figure(2)
        plt.plot(range(x), y, 'b-', label='test acc')
        plt.xticks(range(x))
        plt.legend()
        plt.title("My NN test acc")
        plt.xlabel("epoch")
        plt.ylabel("Accuracy")

        if not os.path.exists("result"):
            os.mkdir("result")
        plt.savefig(f"result/{self.test_acc_file}")
        plt.show()

    def plot_error(self, x, y):
        plt.figure(2)
        plt.plot(range(x), y, 'b-', label='test error')
        plt.xticks(range(x))
        plt.legend()
        plt.title("My NN test error")
        plt.xlabel("epoch")
        plt.ylabel("Error")

        if not os.path.exists("result"):
            os.mkdir("result")
        plt.savefig(f"result/{self.test_error_file}")
        plt.show()

    def train(self, epochs, batch_size, dropout):
        # mess, velocity
        m_w1 = v_w1 = np.zeros(self.W1.shape)
        m_w2 = v_w2 = np.zeros(self.W2.shape)
        m_w3 = v_w3 = np.zeros(self.W3.shape)

        m_b1 = v_b1 = np.zeros(self.b1.shape)
        m_b2 = v_b2 = np.zeros(self.b2.shape)
        m_b3 = v_b3 = np.zeros(self.b3.shape)

        idx = np.arange(self.X_train.shape[0])
        for j in range(epochs):
            np.random.shuffle(idx)

            i = 0
            Loss = 0
            num_batch = 0
            exit_flag = False

            while True:
                t = j + 1
                if i + batch_size >= self.X_train.shape[0]:
                    exit_flag = True
                    batch_X = self.X_train[idx[i:self.X_train.shape[0]]]
                    batch_y = self.y_train[idx[i:self.y_train.shape[0]]]
                    batch_label = self.train_label[idx[i:self.train_label.shape[0]]]
                else:
                    batch_X = self.X_train[idx[i:i + batch_size]]
                    batch_y = self.y_train[idx[i:i + batch_size]]
                    batch_label = self.train_label[idx[i:i + batch_size]]

                if num_batch % 5 == 0 and num_batch != 0:
                    self.calculate_test_acc(batch_X, batch_y)

                # forward
                a1, a2, a3, d_z1, d_z2, m_h1, m_h2 = self.forward(batch_X, dropout=dropout)

                # loss
                loss = self.calculate_error(a3, batch_label)
                if not math.isnan(loss):
                    Loss += loss
                else:
                    pass

                # backward
                grad_w1, grad_w2, grad_w3, grad_b1, grad_b2, grad_b3 = self.backward(a1, a2, a3, d_z1, d_z2, batch_X,
                                                                                     batch_y, m_h1, m_h2)
                # update
                self.W1, m_w1, v_w1 = self.adam_update(self.W1, grad_w1, m_w1, v_w1)
                self.b1, m_b1, v_b1 = self.adam_update(self.b1, grad_b1, m_b1, v_b1)

                self.W2, m_w2, v_w2 = self.adam_update(self.W2, grad_w2, m_w2, v_w2)
                self.b2, m_b2, v_b2 = self.adam_update(self.b2, grad_b2, m_b2, v_b2)

                self.W3, m_w3, v_w3 = self.adam_update(self.W3, grad_w3, m_w3, v_w3)
                self.b3, m_b3, v_b3 = self.adam_update(self.b3, grad_b3, m_b3, v_b3)

                i += batch_size
                num_batch += 1
                if exit_flag:
                    break

            acc = self.calculate_test_acc(self.X_test, self.y_test)
            self.test_acc_list.append(acc)
            self.test_error_list.append(1-acc)

            print(f"Epoch: {j}, Loss: {Loss/num_batch}, test_acc: {acc}")
            self.train_loss_list.append(Loss/num_batch)

        self.plot_loss(epochs, self.train_loss_list)
        self.plot_acc(epochs, self.test_acc_list)
        self.plot_error(epochs, self.test_error_list)



mndata = MNIST('mnist_dataset')

train_img, train_label = mndata.load_training()
train_img, train_label = np.array(train_img), np.array(train_label)
test_img, test_label = mndata.load_testing()
test_img, test_label = np.array(test_img), np.array(test_label)

# print(train_label[0])
model = MyNetwork(784, 512, 512, 10, 0.001, act_f="relu")
model.load_data(train_img, train_label, test_img, test_label)
model.train(epochs=20, batch_size=64, dropout=0.1)
