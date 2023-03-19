import unittest

from testUtils import TestUtils
import pathlib

class LoadAndAddToImageTest(unittest.TestCase):
    # Тестирование функции загрузки и изменения изображения
    def testLoadImage(self):
        import Image
        path = TestUtils.getMainResourcesFolder()
        filename = pathlib.Path(path).joinpath("3.png").resolve()
        image = Image.Image.create(64,64,Image.color_for_name("red"))
        Image.Image.load(image,filename)
        x0 = image.width//4
        y0 = image.height//4
        x1 = image.width//2
        y1 = image.height//2
        image.ellipse(x0, y0, x1, y1, fill=0xFF00FF00)
        path = TestUtils.getResources()
        nameFile = pathlib.Path(path).joinpath("resave_3.png").resolve()
        image.save(nameFile)

        image_2 = Image.Image.create(64, 64, Image.color_for_name("red"))
        Image.Image.load(image_2, filename)
        image_3 = image_2.scale(0.5)
        nameFile = pathlib.Path(path).joinpath("resave_scale_3.png").resolve()
        image_3.save(nameFile)
        
if __name__ == "__main__":
    unittest.main()
