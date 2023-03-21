"""
Применение сверточных слоев,
 стр. 221 Грокаем глубокое обучение
Реализация сети, 28*28 входов, 2 скрытых слоя, 1 выходной слой
Прогнозируем цифры MNIST
Попытка распараллелить вычисления свертки
"""
import concurrent.futures

import numpy as np, sys
#Импорт класса нейронной сети
from common.neuralNetworkClasses.forecastManyManyInOut import ForecastManyManyInOut
from common.timeIt import timeit
from trainClass import TrainClass
from itertools import product

neuralObject = ForecastManyManyInOut()
TrainClassObj = TrainClass()

np.random.seed(1)

import mainVariables as mv
layer_res = 0

def main():
    errorAll, errorAll_test = mainCycle()
    plotResults(errorAll, errorAll_test)

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
def mainCycle():

    errorAll = [[0] * TrainClassObj.labels[0:1].shape[0] for i in range(mv.iterations)]#labels[0:1].shape[0]
    errorAll_test = [[0] * TrainClassObj.labels[0:1].shape[0] for i in range(mv.iterations)]

    for j in range(mv.iterations):
        # Проведение обучения
        trainCycle()

        errorAll[j] = TrainClassObj.correct_cnt / float(len(TrainClassObj.images))

        #Точность теста
        testCycle()

        errorAll_test[j] = TrainClassObj.test_correct_cnt / float(len(TrainClassObj.test_images))

        if (j % 1 == 0):
            sys.stdout.write("\n" + \
                             "I:" + str(j) + \
                             " Test-Acc:" + str(TrainClassObj.test_correct_cnt / float(len(TrainClassObj.test_images))) + \
                             " Train-Acc:" + str(TrainClassObj.correct_cnt / float(len(TrainClassObj.images))))
    return errorAll, errorAll_test

def returnSect(item,layer_res):
    row_start = item[0]
    col_start = item[1]
    k = col_start + row_start * (layer_res.shape[1] - mv.kernel_rows)
    sect = get_image_section(layer_res, row_start,
                             row_start + mv.kernel_rows,
                             col_start,
                             col_start + mv.kernel_cols)
    result = []
    result.append(k)
    result.append(sect)
    return result

def trainFunction(i):
    batch_start, batch_end = ((i * mv.batch_size), ((i + 1) * mv.batch_size))
    layer_0 = TrainClassObj.images[batch_start:batch_end]
    layer_0 = layer_0.reshape(layer_0.shape[0], 28, 28)
    sects = [[] for _ in range((layer_0.shape[1] - mv.kernel_rows)*(layer_0.shape[2] - mv.kernel_cols))]
    temp = list(product(range(layer_0.shape[1] - mv.kernel_rows),range(layer_0.shape[2] - mv.kernel_cols)))
    layer_=[layer_0 for _ in range((layer_0.shape[1] - mv.kernel_rows)*(layer_0.shape[2] - mv.kernel_cols))]
    with concurrent.futures.ProcessPoolExecutor(max_workers=mv.concurrency) as executor:
        for item, result in zip(temp, executor.map(returnSect, temp, layer_)):
            #print('%d is prime: %s' % (item, result[0]))
            sects[result[0]] = result[1]
    executor.shutdown()
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

def testFunction(i):
    layer_0 = TrainClassObj.test_images[i:i + 1]
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

    kernel_output = flattened_input.dot(TrainClassObj.kernels)
    layer_1 = tanh(kernel_output.reshape(es[0], -1))
    layer_2 = np.dot(layer_1, TrainClassObj.weights_1_2)

    TrainClassObj.test_correct_cnt += int(np.argmax(layer_2) ==
                                          np.argmax(TrainClassObj.test_labels[i:i + 1]))

@timeit("Train Cycle")
def trainCycle():
    for i in range(int(len(TrainClassObj.images) / mv.batch_size)):
        trainFunction(i)

@timeit("Test Cycle")
def testCycle():
    for i in range(len(TrainClassObj.test_images)):
        testFunction(i)

def plotResults(errorAll, errorAll_test):
    name = 'Сверточная сеть обучение'
    neuralObject.plot_accuracy(mv.iterations, errorAll,name)
    name = 'Сверточная сеть тест'
    neuralObject.plot_accuracy(mv.iterations, errorAll_test,name)

if __name__ == '__main__':
    main()