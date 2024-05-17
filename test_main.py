import unittest
import numpy as np
from PIL import Image
from main import load_model, preprocess_image, print_predictions


class TestImageClassificationApp(unittest.TestCase):
    def setUp(self):
        self.model = load_model()

    
    def test_load_model(self):
        self.assertIsNotNone(self.model, "Модель не загружена")


    def test_preprocess_image(self):
        img = Image.new('RGB', (224, 224))
        preprocessed_img = preprocess_image(img)
        self.assertIsInstance(preprocessed_img, np.ndarray, "Изображение не было правильно предварительно обработано")


    def test_print_predictions(self):
    # Создаем тестовые предсказания с формой (1, 1000)
        preds = np.random.rand(1, 1000)
        print_predictions(preds)
        # Здесь мы предполагаем, что функция print_predictions выводит результаты в stdout,
        # поэтому мы не можем напрямую проверить вывод. Вместо этого мы можем проверить, что функция не вызывает исключений.
        pass


if __name__ == '__main__':
    unittest.main()
