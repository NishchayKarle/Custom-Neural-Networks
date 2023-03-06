from mnist import MNIST
import numpy as np


class Neural_Network:
    def __init__(self, nl, nh, ne, nb):
        self.alpha = 0.1
        self.epsilon = 0.009

        self.input_neurons = 784
        self.output_neurons = 10

        self.nl = nl
        self.nh = nh
        self.ne = ne
        self.nb = nb

        self.num_layers = 2 + self.nl

        self.il_weights = np.random.rand(self.nh, self.input_neurons)
        self.hl_weights = np.random.rand(self.nl - 1, self.nh, self.nh)
        self.ol_weights = np.random.rand(self.output_neurons, self.nh)

        self.hl_bias = np.random.rand(self.nl, self.nh)
        self.ol_bias = np.random.rand(1, self.output_neurons)

        self.hl_z = np.zeros(self.nl * self.nh).reshape(self.nl, self.nh)
        self.ol_z = np.zeros(self.output_neurons).reshape(
            1, self.output_neurons)

        self.il_activation = np.zeros(
            self.input_neurons).reshape(1, self.input_neurons)
        self.hl_activation = np.zeros(
            self.nl * self.nh).reshape(self.nl, self.nh)
        self.ol_activation = np.zeros(
            self.output_neurons).reshape(1, self.output_neurons)

        self.hl_error = np.zeros(self.nl * self.nh).reshape(self.nl, self.nh)
        self.ol_error = np.zeros(self.output_neurons).reshape(
            1, self.output_neurons)

    def sigmoid(self, val):
        return 1/(1 + np.exp(-val))

    def dsigmoid(self, val):
        return self.sigmoid(val) * (1 - self.sigmoid(val))

    def feed_forward(self):
        # print(self.il_weights.shape, self.il_activation.shape)
        self.hl_z[0] = np.matmul(self.il_weights,
                                 self.il_activation) + self.hl_bias[0]
        self.hl_activation[0] = self.sigmoid(self.hl_z[0])

        if self.nl > 1:
            for l in range(1, self.nl):
                self.hl_z[l] = np.matmul(self.hl_weights[l - 1],
                                         self.hl_activation[l-1]) + self.hl_bias[l]

                self.hl_activation[l] = self.sigmoid(self.hl_z[l])

        self.ol_z = np.matmul(
            self.ol_weights, self.hl_activation[-1]) + self.ol_bias
        self.ol_activation = self.sigmoid(self.ol_z)

    def cal_output_error(self, val, loop):
        y = np.zeros(10)
        y[int(val)] = 1
        # print(val, y)
        self.ol_error = (self.ol_activation - y)
        # if loop % 100 == 0:
        #     print(np.linalg.norm(self.ol_error))

    def back_propogate(self):
        self.hl_error[self.nl - 1] = np.matmul(self.ol_error, self.ol_weights,
                                               ) * self.dsigmoid(self.hl_z[-1])

        for l in range(self.nl - 2, -1, -1):
            self.hl_error[l] = np.matmul(self.hl_error[l + 1],
                                         self.hl_weights[l]) * self.dsigmoid(self.hl_z[l])

    def gradient_descent(self, images, y, loop):
        ol_dweight_sum = np.zeros(
            self.output_neurons * self.nh).reshape(self.output_neurons, self.nh)
        hl_dweight_sum = np.zeros(
            (self.nl - 1) * self.nh * self.nh).reshape(self.nl - 1, self.nh, self.nh)
        il_weight_sum = np.zeros(
            self.nh * self.input_neurons).reshape(self.nh, self.input_neurons)

        ol_dbias_sum = np.zeros(self.output_neurons).reshape(
            1, self.output_neurons)
        hl_dbias_sum = np.zeros(self.nl * self.nh).reshape(self.nl, self.nh)

        for i in range(self.nb):
            self.il_activation = images[i]
            # print(self.il_activation)
            self.feed_forward()
            self.cal_output_error(y[i], loop)
            self.back_propogate()

            ol_dweight_sum += np.matmul(self.ol_error.T,
                                        self.hl_activation[-1].reshape(1, self.nh))
            for l in range(self.nl - 2, 1, -1):
                hl_dweight_sum[l] += np.matmul(self.hl_error[l].T,
                                               self.hl_activation[l-1])

            # print(self.hl_error[0].T.reshape(self.nh, 1).shape, self.il_activation.reshape(
                # 1, self.input_neurons).shape)
            il_weight_sum += np.matmul(
                self.hl_error[0].T.reshape(self.nh, 1), self.il_activation.reshape(1, self.input_neurons))

            ol_dbias_sum += self.ol_error
            hl_dbias_sum += self.hl_error

        # print(np.linalg.norm(ol_dbias_sum))
        self.ol_weights -= (ol_dweight_sum * self.alpha / self.nb)
        self.hl_weights -= (hl_dweight_sum * self.alpha / self.nb)
        self.il_weights -= (il_weight_sum * self.alpha / self.nb)

        self.ol_bias -= (ol_dbias_sum * self.alpha / self.nb)
        self.hl_bias -= (hl_dbias_sum * self.alpha / self.nb)

        # print(ol_dweight_sum * self.alpha)
        # print(ol_dbias_sum * self.alpha)
        flag = True
        flag &= (np.abs(self.ol_weights) < self.epsilon).all()
        flag &= (np.abs(self.hl_weights) < self.epsilon).all()
        flag &= (np.abs(self.il_weights) < self.epsilon).all()
        flag &= (np.abs(self.ol_bias) < self.epsilon).all()
        flag &= (np.abs(self.hl_bias) < self.epsilon).all()

        return flag

    def train(self, images, y):
        print("STARTED TRAINGING")
        n = len(images)

        for j in range(self.ne):
            indexes = np.random.choice(n, n, replace=False)
            for i in range(int(n/self.nb)):
                images_batch = images[indexes[self.nb *
                                              i: (i + 1) * self.nb]]
                y_batch = y[indexes[self.nb * i: (i + 1) * self.nb]]

                self.gradient_descent(images_batch, y_batch, j)
                # print(*self.ol_error)

                print(np.argmax(self.ol_activation), y_batch[i])

    def test(self, images, y):
        print("STARTED TESTING")
        correct = 0
        total = len(images)
        for i, image in enumerate(images):
            self.il_activation = image
            self.feed_forward()
            # print(np.argmax(self.ol_activation.T), y[i])
            # print("xxxxx", self.ol_activation, y[i])
            # print("asdhkasdc")
            if np.argmax(self.ol_activation.T) == int(y[i]):
                correct += 1

        print("ACC: {}% \n".format(100 * correct/total))


data = MNIST('data')
X, y = data.load_training()
X = np.array(X, dtype='float32')
y = np.array(y, dtype='float32')
# print(y[0])

N = Neural_Network(1, 10, 500, 100)
N.train(X[:100], y[:100])

X_test, y_test = data.load_testing()
X_test = np.array(X_test, dtype='float32')
y_test = np.array(y_test, dtype='float32')

N.test(X_test[:10], y_test[:10])
