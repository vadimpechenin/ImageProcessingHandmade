import unittest
from testUtils import TestUtils
import pathlib
import os


class MultiprocessingTest(unittest.TestCase):
    # Тестирование функции многопроцессорного выполнения программ python
    def testOneProcess(self):
        path = TestUtils.getMultiprocessing()
        filename = str(pathlib.Path(path).joinpath("imagescale-s.py").resolve())
        os.system("python " + filename)


if __name__ == "__main__":
    unittest.main()
