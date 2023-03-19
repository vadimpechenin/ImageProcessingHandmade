import unittest

from testUtils import TestUtils
import pathlib

class QualityIdentificationNNTest(unittest.TestCase):
    # Тестирование функции создания и записи в БД новой детали
    def testCreateImage(self):
        import Image
        image = Image.Image.create(64,64,Image.color_for_name("red"))
        path = TestUtils.getResources()
        nameFile = pathlib.Path(path).joinpath("red_64x64.xpm").resolve()
        image.save(nameFile)

if __name__ == "__main__":
    unittest.main()
