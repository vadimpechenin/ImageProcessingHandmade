import unittest
from testUtils import TestUtils
import pathlib
import os
from datetime import datetime


class MultiprocessingTest(unittest.TestCase):
    # Тестирование функции многопроцессорного выполнения программ python
    def testOneProcess(self):
        path = TestUtils.getMultiprocessingSimple()

        filename = str(pathlib.Path(path).joinpath("exampleSimple.py").resolve())
        os.system("python " + filename)

        print("***********")
        filename = str(pathlib.Path(path).joinpath("exampleOfConcurrent.py").resolve())
        os.system("python " + filename)
if __name__ == "__main__":
    unittest.main()
