import numpy as np
import multiprocessing

alpha, iterations = (2, 2)
pixels_per_image, num_labels = (784, 10)
batch_size = 128

input_rows = 28
input_cols = 28

kernel_rows = 3
kernel_cols = 3
num_kernels = 16

hidden_size = ((input_rows - kernel_rows) *
               (input_cols - kernel_cols)) * num_kernels

kernels = 0.02 * np.random.random((kernel_rows * kernel_cols,
                                   num_kernels)) - 0.01

weights_1_2 = 0.2 * np.random.random((hidden_size,
                                      num_labels)) - 0.1

concurrency = multiprocessing.cpu_count()