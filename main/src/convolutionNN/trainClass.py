import numpy as np
np.random.seed(1)

from keras.datasets import mnist
import mainVariables as mv

class TrainClass():

    def __init__(self):
        self.kernels = 0.02 * np.random.random((mv.kernel_rows * mv.kernel_cols,
                                           mv.num_kernels)) - 0.01

        self.weights_1_2 = 0.2 * np.random.random((mv.hidden_size,
                                              mv.num_labels)) - 0.1

        self.correct_cnt = 0
        self.test_correct_cnt = 0
        self.images = None
        self.labels = None
        self.test_images = None
        self.test_labels = None
        self.getTestTrainSample()

    def getTestTrainSample(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        self.images, self.labels = (x_train[0:1000].reshape(1000, 28 * 28) / 255,
                        y_train[0:1000])

        one_hot_labels = np.zeros((len(self.labels), 10))
        for i, l in enumerate(self.labels):
            one_hot_labels[i][l] = 1
        self.labels = one_hot_labels

        self.test_images, self.test_labels = (x_test[0:100].reshape(100, 28 * 28) / 255,
                        y_test[0:100])

        #test_images = x_test.reshape(len(x_test), 28 * 28) / 255
        one_hot_labels = np.zeros((len(y_test), 10))
        for i, l in enumerate(self.test_labels):
            one_hot_labels[i][l] = 1
        self.test_labels = one_hot_labels