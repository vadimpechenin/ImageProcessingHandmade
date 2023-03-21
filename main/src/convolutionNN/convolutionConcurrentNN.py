"""
Применение сверточных слоев,
 стр. 221 Грокаем глубокое обучение
Реализация сети, 28*28 входов, 2 скрытых слоя, 1 выходной слой
Прогнозируем цифры MNIST
"""
import copy
import concurrent.futures

import numpy as np, sys
#Импорт класса нейронной сети
from common.neuralNetworkClasses.forecastManyManyInOut import ForecastManyManyInOut
from common.timeIt import timeit

neuralObject = ForecastManyManyInOut()

np.random.seed(1)

from tensorflow.keras.datasets import mnist
import mainVariables as mv

def main():

    images, labels, test_images, test_labels = getTestTrainSample()
    errorAll, errorAll_test = mainCycle(images, labels, test_images, test_labels)
    plotResults(errorAll, errorAll_test)

def getTestTrainSample():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    images, labels = (x_train[0:1000].reshape(1000, 28 * 28) / 255,
                    y_train[0:1000])

    one_hot_labels = np.zeros((len(labels), 10))
    for i, l in enumerate(labels):
        one_hot_labels[i][l] = 1
    labels = one_hot_labels

    test_images, test_labels = (x_test[0:100].reshape(100, 28 * 28) / 255,
                    y_test[0:100])

    #test_images = x_test.reshape(len(x_test), 28 * 28) / 255
    one_hot_labels = np.zeros((len(y_test), 10))
    for i, l in enumerate(test_labels):
        one_hot_labels[i][l] = 1
    test_labels = one_hot_labels
    return images, labels, test_images, test_labels

def tanh(x):
    return np.tanh(x)


def tanh2deriv(output):
    return 1 - (output ** 2)


def softmax(x):
    temp = np.exp(x)
    return temp / np.sum(temp, axis=1, keepdims=True)

def get_image_section(layer, row_from, row_to, col_from, col_to):
    #Выделяем часть изображения для работы с ней
    section = layer[:, row_from:row_to, col_from:col_to]
    return section.reshape(-1, 1, row_to - row_from, col_to - col_from)

@timeit("Main Cycle")
def mainCycle(images, labels, test_images, test_labels):

    errorAll = [[0] * labels[0:1].shape[0] for i in range(mv.iterations)]#labels[0:1].shape[0]
    errorAll_test = [[0] * labels[0:1].shape[0] for i in range(mv.iterations)]
    weights_1_2 = copy.deepcopy(mv.weights_1_2)
    kernels = copy.deepcopy(mv.kernels)

    for j in range(mv.iterations):
        correct_cnt = 0

        # Проведение обучения
        weights_1_2, kernels, correct_cnt = trainCycle(weights_1_2, kernels, correct_cnt, images, labels)

        errorAll[j] = correct_cnt / float(len(images))

        test_correct_cnt = 0

        #Точность теста
        test_correct_cnt = testCycle(test_images, test_labels, kernels, weights_1_2, test_correct_cnt)

        errorAll_test[j] = test_correct_cnt / float(len(test_images))

        if (j % 1 == 0):
            sys.stdout.write("\n" + \
                             "I:" + str(j) + \
                             " Test-Acc:" + str(test_correct_cnt / float(len(test_images))) + \
                             " Train-Acc:" + str(correct_cnt / float(len(images))))
    return errorAll, errorAll_test

def trainFunction(i,weights_1_2,kernels,correct_cnt,images,labels):
    batch_start, batch_end = ((i * mv.batch_size), ((i + 1) * mv.batch_size))
    layer_0 = images[batch_start:batch_end]
    layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)

    sects = list()
    for row_start in range(layer_0.shape[1] - mv.kernel_rows):
        for col_start in range(layer_0.shape[2] - mv.kernel_cols):
            sect = get_image_section(layer_0,
                                     row_start,
                                     row_start + mv.kernel_rows,
                                     col_start,
                                     col_start + mv.kernel_cols)
            sects.append(sect)

    expanded_input = np.concatenate(sects, axis=1)
    es = expanded_input.shape
    flattened_input = expanded_input.reshape(es[0] * es[1], -1)

    kernel_output = flattened_input.dot(kernels)
    layer_1 = tanh(kernel_output.reshape(es[0], -1))
    dropout_mask = np.random.randint(2, size=layer_1.shape)
    layer_1 *= dropout_mask * 2
    layer_2 = softmax(np.dot(layer_1, mv.weights_1_2))

    for k in range(mv.batch_size):
        labelset = labels[batch_start + k:batch_start + k + 1]
        _inc = int(np.argmax(layer_2[k:k + 1]) ==
                   np.argmax(labelset))
        correct_cnt += _inc

    layer_2_delta = (labels[batch_start:batch_end] - layer_2) \
                    / (mv.batch_size * layer_2.shape[0])
    layer_1_delta = layer_2_delta.dot(mv.weights_1_2.T) * \
                    tanh2deriv(layer_1)
    layer_1_delta *= dropout_mask
    weights_1_2 += mv.alpha * layer_1.T.dot(layer_2_delta)
    l1d_reshape = layer_1_delta.reshape(kernel_output.shape)
    k_update = flattened_input.T.dot(l1d_reshape)
    kernels -= mv.alpha * k_update

    return weights_1_2,kernels, correct_cnt

def testFunction(i,test_images,test_labels, kernels, weights_1_2, test_correct_cnt):
    layer_0 = test_images[i:i + 1]
    layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)

    sects = list()
    for row_start in range(layer_0.shape[1] - mv.kernel_rows):
        for col_start in range(layer_0.shape[2] - mv.kernel_cols):
            sect = get_image_section(layer_0,
                                     row_start,
                                     row_start + mv.kernel_rows,
                                     col_start,
                                     col_start + mv.kernel_cols)
            sects.append(sect)

    expanded_input = np.concatenate(sects, axis=1)
    es = expanded_input.shape
    flattened_input = expanded_input.reshape(es[0] * es[1], -1)

    kernel_output = flattened_input.dot(kernels)
    layer_1 = tanh(kernel_output.reshape(es[0], -1))
    layer_2 = np.dot(layer_1, weights_1_2)

    test_correct_cnt += int(np.argmax(layer_2) ==
                            np.argmax(test_labels[i:i + 1]))

    return test_correct_cnt

@timeit("Train Cycle")
def trainCycle(weights_1_2, kernels, correct_cnt, images, labels):
    for i in range(int(len(images) / mv.batch_size)):
        weights_1_2, kernels, correct_cnt = trainFunction(i, weights_1_2, kernels, correct_cnt, images, labels)
    return weights_1_2, kernels, correct_cnt

@timeit("Test Cycle")
def testCycle(test_images, test_labels, kernels, weights_1_2, test_correct_cnt):
    with concurrent.futures.ProcessPoolExecutor(max_workers=mv.concurrency) as executor:
        for number, test_correct_cnt in zip(range(len(test_images)), executor.map(testFunction, range(len(test_images)), test_images, test_labels, kernels, weights_1_2, test_correct_cnt)):
            print('%d is prime: %s' % (number, test_correct_cnt))
    return test_correct_cnt
    """for i in range(len(test_images)):
        test_correct_cnt = testFunction(i, test_images, test_labels, kernels, weights_1_2, test_correct_cnt)"""
    return test_correct_cnt

def plotResults(errorAll, errorAll_test):
    name = 'Сверточная сеть обучение'
    neuralObject.plot_accuracy(mv.iterations, errorAll,name)
    name = 'Сверточная сеть тест'
    neuralObject.plot_accuracy(mv.iterations, errorAll_test,name)

if __name__ == '__main__':
    main()