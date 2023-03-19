import unittest

from testUtils import TestUtils
import pathlib

class CreateNewImageTest(unittest.TestCase):
    # Тестирование функции создания и записи в БД новой детали
    def testCreateImage(self):
        import Image
        image = Image.Image.create(64,64,Image.color_for_name("red"))
        path = TestUtils.getResources()
        nameFile = pathlib.Path(path).joinpath("red_64x64.xpm").resolve()
        image.save(nameFile)

        image_2 = Image.Image.create(1024, 1024, Image.color_for_name("red"))
        x0 = 30
        y0 = 30
        x1 = 600
        y1 = 600
        image_2.ellipse(x0, y0, x1, y1, fill=0xFF00FF00)

        path = TestUtils.getResources()
        nameFile = pathlib.Path(path).joinpath("red_ellipse_1024x1024.png").resolve()
        image_2.save(nameFile)

        image_3 = image_2.scale(0.5)
        nameFile = pathlib.Path(path).joinpath("red_ellipse_512x512.png").resolve()
        image_3.save(nameFile)

if __name__ == "__main__":
    unittest.main()
