import unittest
import io
import numpy as np
from PIL import Image
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocessinput, decodepredictions
import streamlit as st

from main.py import app

from streamlit.testing.v1 import AppTest

at = AppTest.from_file("main.py")

def test_load_image():
    at = AppTest.from_file("main.py")
    at.run(timeout=10) 



#class TestStreamlitApp(unittest.TestCase):

    #def setUp(self):
        self.model = app.loadmodel()

    #def testload_model(self):
        # Проверяем, что модель успешно загружается
        self.assertIsNotNone(self.model)

    #def test_preprocess_image(self):
        # Создаем тестовое изображение
        test_img = Image.new('RGB', (224, 224), color = (73, 109, 137))
        preprocessed_img = app.preprocess_image(test_img)
        # Проверяем, что изображение было правильно преобразовано
        self.assertEqual(preprocessed_img.shape, (1, 224, 224, 3))

    #def test_decode_predictions(self):
        # Создаем тестовые предсказания
        test_preds = np.array([[0.1, 0.2, 0.7]])
        decoded_preds = app.decode_predictions(test_preds, top=3)
        # Проверяем, что предсказания были правильно декодированы
        self.assertEqual(len(decoded_preds), 3)
        self.assertIn('airplane', decoded_preds[0])

    #def test_print_predictions(self):
        # Создаем тестовые предсказания
        test_preds = np.array([[0.1, 0.2, 0.7]])
        app.print_predictions(test_preds)
        # Здесь мы проверяем вывод, но это может быть сложно без GUI
        # Вместо этого можно проверить, что функция вызывается без ошибок

#if __name__ == '__main__':
    unittest.main()



   
    

