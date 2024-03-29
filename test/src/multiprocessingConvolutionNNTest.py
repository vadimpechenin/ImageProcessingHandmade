import unittest
from testUtils import TestUtils
import pathlib
import os
from datetime import datetime


class MultiprocessingTest(unittest.TestCase):
    # Тестирование функции многопроцессорного выполнения работы сверточной нейронной сети
    def testProcessOfConvolution(self):
        path = TestUtils.getMultiprocessingConvolutionNN()

        filename = str(pathlib.Path(path).joinpath("convolutionNN.py").resolve())
        os.system("python " + filename)

        print("***********")
        filename = str(pathlib.Path(path).joinpath("convolutionConcurrentNN.py").resolve())
        os.system("python " + filename)

if __name__ == "__main__":
    unittest.main()
