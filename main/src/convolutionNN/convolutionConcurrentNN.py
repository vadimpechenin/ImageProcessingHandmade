"""
Применение сверточных слоев,
 стр. 221 Грокаем глубокое обучение
Реализация сети, 28*28 входов, 2 скрытых слоя, 1 выходной слой
Попытка распараллелить вычисления тестовых значений (для начала)
"""
import concurrent.futures

import copy

import numpy as np, sys
#Импорт класса нейронной сети
from common.neuralNetworkClasses.forecastManyManyInOut import ForecastManyManyInOut
from common.timeIt import timeit, timeitNoPrint
from trainClass import TrainClass
from itertools import product

neuralObject = ForecastManyManyInOut()
TrainClassObj = TrainClass()

np.random.seed(1)

from keras.datasets import mnist
import mainVariables as mv

def main():

    #getTestTrainSample()
    errorAll, errorAll_test = mainCycle()
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

    TrainClassObj.images = images
    TrainClassObj.labels = labels
    TrainClassObj.test_images = test_images
    TrainClassObj.test_labels = test_labels
    images = None
    g = 0
    #return images, labels, test_images, test_labels

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

@timeitNoPrint("Main Cycle")
def mainCycle():

    errorAll = [[0] * TrainClassObj.labels[0:1].shape[0] for i in range(mv.iterations)]#labels[0:1].shape[0]
    errorAll_test = [[0] * TrainClassObj.labels[0:1].shape[0] for i in range(mv.iterations)]
    #weights_1_2 = copy.deepcopy(mv.weights_1_2)
    #kernels = copy.deepcopy(mv.kernels)

    for j in range(mv.iterations):
        # Проведение обучения
        trainCycle()#weights_1_2, kernels, correct_cnt, images, labels

        errorAll[j] = TrainClassObj.correct_cnt / float(len(TrainClassObj.images))

        #test_correct_cnt = 0

        #Точность теста
        TrainClassObj.test_correct_cnt = testCycle()#test_images, test_labels, kernels, weights_1_2, test_correct_cnt

        errorAll_test[j] = TrainClassObj.test_correct_cnt / float(len(TrainClassObj.test_images))

        if (j % 1 == 0):
            sys.stdout.write("\n" + \
                             "I:" + str(j) + \
                             " Test-Acc:" + str(TrainClassObj.test_correct_cnt / float(len(TrainClassObj.test_images))) + \
                             " Train-Acc:" + str(TrainClassObj.correct_cnt / float(len(TrainClassObj.images))))
    return errorAll, errorAll_test

def returnSect(item, layer_0, sects):
    row_start = item[0]
    col_start = item[1]
    k = col_start + row_start * (layer_0.shape[1] - mv.kernel_rows)
    sect = get_image_section(layer_0,
                             row_start,
                             row_start + mv.kernel_rows,
                             col_start,
                             col_start + mv.kernel_cols)
    sects[k] = sect
    return k

def trainFunction(i):#,weights_1_2,kernels,correct_cnt,images,labels
    batch_start, batch_end = ((i * mv.batch_size), ((i + 1) * mv.batch_size))
    layer_0 = TrainClassObj.images[batch_start:batch_end]
    layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)

    sects = [[] for _ in range((layer_0.shape[1] - mv.kernel_rows) * (layer_0.shape[2] - mv.kernel_cols))]
    temp = list(product(range(layer_0.shape[1] - mv.kernel_rows), range(layer_0.shape[2] - mv.kernel_cols)))

    for item in temp:
        k = returnSect(item, layer_0, sects)

    expanded_input = np.concatenate(sects, axis=1)
    es = expanded_input.shape
    flattened_input = expanded_input.reshape(es[0] * es[1], -1)

    kernel_output = flattened_input.dot(TrainClassObj.kernels)
    layer_1 = tanh(kernel_output.reshape(es[0], -1))
    dropout_mask = np.random.randint(2, size=layer_1.shape)
    layer_1 *= dropout_mask * 2
    layer_2 = softmax(np.dot(layer_1, TrainClassObj.weights_1_2))

    for k in range(mv.batch_size):
        labelset = TrainClassObj.labels[batch_start + k:batch_start + k + 1]
        _inc = int(np.argmax(layer_2[k:k + 1]) ==
                   np.argmax(labelset))
        TrainClassObj.correct_cnt += _inc

    layer_2_delta = (TrainClassObj.labels[batch_start:batch_end] - layer_2) \
                    / (mv.batch_size * layer_2.shape[0])
    layer_1_delta = layer_2_delta.dot(TrainClassObj.weights_1_2.T) * \
                    tanh2deriv(layer_1)
    layer_1_delta *= dropout_mask
    TrainClassObj.weights_1_2 += mv.alpha * layer_1.T.dot(layer_2_delta)
    l1d_reshape = layer_1_delta.reshape(kernel_output.shape)
    k_update = flattened_input.T.dot(l1d_reshape)
    TrainClassObj.kernels -= mv.alpha * k_update

    #return weights_1_2,kernels, correct_cnt

def testFunction(i):#,test_images,test_labels, kernels, weights_1_2, test_correct_cnt
    layer_0 = TrainClassObj.test_images[i:i + 1]
    layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)

    sects = [[] for _ in range((layer_0.shape[1] - mv.kernel_rows) * (layer_0.shape[2] - mv.kernel_cols))]
    temp = list(product(range(layer_0.shape[1] - mv.kernel_rows), range(layer_0.shape[2] - mv.kernel_cols)))

    for item in temp:
        k = returnSect(item, layer_0, sects)

    expanded_input = np.concatenate(sects, axis=1)
    es = expanded_input.shape
    flattened_input = expanded_input.reshape(es[0] * es[1], -1)

    kernel_output = flattened_input.dot(TrainClassObj.kernels)
    layer_1 = tanh(kernel_output.reshape(es[0], -1))
    layer_2 = np.dot(layer_1, TrainClassObj.weights_1_2)

    """TrainClassObj.test_correct_cnt += int(np.argmax(layer_2) ==
                            np.argmax(TrainClassObj.test_labels[i:i + 1]))"""
    return int(np.argmax(layer_2) ==
                            np.argmax(TrainClassObj.test_labels[i:i + 1]))
    #return test_correct_cnt

@timeitNoPrint("Train Cycle")
def trainCycle():
    for i in range(int(len(TrainClassObj.images) / mv.batch_size)):
        trainFunction(i)

@timeitNoPrint("Test Cycle")
def testCycle():
    test_correct_cnt = 0
    i_list=[i for i in range(len(TrainClassObj.test_images))]
    with concurrent.futures.ProcessPoolExecutor(max_workers=mv.concurrency) as executor:
        for item, result in zip(i_list, executor.map(testFunction, i_list)):
            test_correct_cnt += result
    executor.shutdown()
    return test_correct_cnt
def plotResults(errorAll, errorAll_test):
    name = 'Сверточная сеть обучение'
    neuralObject.plot_accuracy(mv.iterations, errorAll,name)
    name = 'Сверточная сеть тест'
    neuralObject.plot_accuracy(mv.iterations, errorAll_test,name)

if __name__ == '__main__':
    main()