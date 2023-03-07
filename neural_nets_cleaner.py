from mnist import MNIST
import numpy as np


class Neural_Network:
    def __init__(self, nl, nh, ne, nb):
        self.alpha = 1
        self.il = 784
        self.ol = 10

        self.nl = nl
        self.nh = nh
        self.ne = ne
        self.nb = nb

        self.tl = self.nl + 2

        self.weights = []
        self.weights.append(np.random.rand(self.nh, self.il))
        for _ in range(self.nl - 1):
            self.weights.append(np.random.rand(self.nh, self.nh))
        self.weights.append(np.random.rand(self.ol, self.nh))

        self.bias = []
        self.bias.append(np.random.rand(self.il, 1))
        for _ in range(self.nl):
            self.bias.append((np.random.rand(self.nh, 1)))
        self.bias.append(np.zeros((self.ol, 1)))

        self.z = []
        self.z.append(np.random.rand(self.il, 1))
        for _ in range(self.nl):
            self.z.append((np.random.rand(self.nh, 1)))
        self.z.append(np.zeros((self.ol, 1)))

        self.a = []
        self.a.append(np.random.rand(self.il, 1))
        for _ in range(self.nl):
            self.a.append((np.random.rand(self.nh, 1)))
        self.a.append(np.zeros((self.ol, 1)))

        self.e = []
        self.e.append(np.random.rand(self.il, 1))
        for _ in range(self.nl):
            self.e.append((np.random.rand(self.nh, 1)))
        self.e.append(np.zeros((self.ol, 1)))

    def sigmoid(self, val):
        return 1/(1 + np.exp(-val))

    def dsigmoid(self, val):
        return self.sigmoid(val) * (1 - self.sigmoid(val))

    def feed_forward(self, input_x, test=True):
        self.a[0] = input_x.reshape(self.il, 1)

        for l in range(1, self.tl):
            self.z[l] = np.matmul(
                self.weights[l-1], self.a[l-1]) + self.bias[l]
            self.a[l] = self.sigmoid(self.z[l])

    def output_error(self, val):
        y = np.zeros(10)
        y[val] = 1
        y.resize(10, 1)

        self.e[self.tl - 1] = (self.a[self.tl - 1] - y) * \
            self.dsigmoid(self.z[self.tl - 1])

    def backpropogate(self):
        for l in range(self.tl - 2, 0, -1):
            self.e[l] = np.matmul(
                self.weights[l].T, self.e[1]) * self.dsigmoid(self.z[l])

    def gradient_descent(self, input_x, input_y):
        dweights = []
        dweights.append(np.zeros((self.nh, self.il)))
        for _ in range(self.nl - 1):
            dweights.append(np.zeros((self.nh, self.nh)))
        dweights.append(np.zeros((self.ol, self.nh)))

        dbias = []
        dbias.append(np.random.rand(self.il, 1))
        for _ in range(self.nl):
            dbias.append((np.random.rand(self.nh, 1)))
        dbias.append(np.zeros((self.ol, 1)))

        for i in range(self.nb):
            self.feed_forward(input_x[i], False)
            self.output_error(input_y[i])
            self.backpropogate()

            for l in range(self.tl - 1, 0, -1):
                dweights[l-1] += np.matmul(self.e[l], self.a[l-1].T)
                dbias[l] += self.e[l]

        for l in range(self.tl - 1):
            dweights[l] *= (self.alpha / self.nb)
            self.weights[l] -= dweights[l]

            dbias[l + 1] *= (self.alpha / self.nb)
            self.bias[l + 1] -= dbias[l + 1]

    def train(self, images, y):
        print("STARTED TRAINGING")
        n = len(images)

        for j in range(self.ne):
            indexes = np.random.choice(n, n, replace=False)
            for i in range(int(n/self.nb)):
                images_batch = images[indexes[self.nb *
                                              i: (i + 1) * self.nb]]
                y_batch = y[indexes[self.nb * i: (i + 1) * self.nb]]

                self.gradient_descent(images_batch, y_batch)

    def test(self, images, y):
        print("STARTED TESTING")
        correct = 0
        total = len(images)
        for i, image in enumerate(images):
            self.feed_forward(image, test=True)
            if np.argmax(self.a[-1]) == y[i]:
                correct += 1

        print("ACC: {}% \n".format(100 * correct/total))


data = MNIST('data')
X, y = data.load_training()
X = np.array(X, dtype='float128')
y = np.array(y)
X /= 255

N = Neural_Network(3, 10, 2, 1)
N.train(X[:1000], y[:1000])

X_test, y_test = data.load_testing()
X_test = np.array(X_test, dtype='float128')
y_test = np.array(y_test)

X_test /= 255

N.test(X_test[:100], y_test[:100])
