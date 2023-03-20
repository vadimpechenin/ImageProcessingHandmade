import unittest
from random import random

from testUtils import TestUtils
import pathlib
import Image


class CreateNewImageTest(unittest.TestCase):
    # Тестирование функции создания новых деталей
    def testCreateImage(self):
        path = TestUtils.getMainResourcesXPMFolder()
        for j in range(50):
            image = Image.Image.create(256,256,Image.color_for_name("red"))
            x0 = round(random()*30)
            y0 = round(random()*30)
            x1 = round(31+random()*200)
            y1 = round(31+random()*200)
            image.ellipse(x0, y0, x1, y1, fill=0xFF00FF00)
            nameFile = pathlib.Path(path).joinpath("im_" + str(j) + ".xpm").resolve()
            image.save(nameFile)

if __name__ == "__main__":
    unittest.main()
