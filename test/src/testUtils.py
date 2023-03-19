import pathlib
import unittest

from common.commonUtils import CommonUtils


class TestUtils(unittest.TestCase):
    @staticmethod
    def getTestFolder():
        solutionFolder = CommonUtils.getSolutionFolder()
        return pathlib.Path(solutionFolder).joinpath("ImageProcessingHandmade").joinpath("test").resolve()

    @staticmethod
    def getResources():
        solutionFolder = CommonUtils.getSolutionFolder()
        return pathlib.Path(solutionFolder).joinpath("ImageProcessingHandmade").joinpath("test").joinpath("resultsData").resolve()

    @staticmethod
    def getMainResourcesFolder():
        return CommonUtils.getMainResourcesFolder()


if __name__ == "__main__":
    unittest.main()
