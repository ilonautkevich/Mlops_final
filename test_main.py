#import unittest
#import io
#import numpy as np
#from PIL import Image
#from tensorflow.keras.applications import EfficientNetB0
#from tensorflow.keras.preprocessing import image
#from tensorflow.keras.applications.efficientnet import preprocessinput, decodepredictions
#import streamlit as st

#from main.py import app

#from streamlit.testing.v1 import AppTest

#at = AppTest.from_file("main.py")

#def test_load_image():
    #at = AppTest.from_file("main.py")
    #at.run(timeout=10) 



#class TestStreamlitApp(unittest.TestCase):

    #def setUp(self):
        #self.model = app.loadmodel()

    #def testload_model(self):
        # Проверяем, что модель успешно загружается
        #self.assertIsNotNone(self.model)

    #def test_preprocess_image(self):
        # Создаем тестовое изображение
        #test_img = Image.new('RGB', (224, 224), color = (73, 109, 137))
        #preprocessed_img = app.preprocess_image(test_img)
        # Проверяем, что изображение было правильно преобразовано
        #self.assertEqual(preprocessed_img.shape, (1, 224, 224, 3))

    #def test_decode_predictions(self):
        # Создаем тестовые предсказания
        #test_preds = np.array([[0.1, 0.2, 0.7]])
        #decoded_preds = app.decode_predictions(test_preds, top=3)
        # Проверяем, что предсказания были правильно декодированы
        #self.assertEqual(len(decoded_preds), 3)
        #self.assertIn('airplane', decoded_preds[0])

    #def test_print_predictions(self):
        # Создаем тестовые предсказания
        #test_preds = np.array([[0.1, 0.2, 0.7]])
        #app.print_predictions(test_preds)
        # Здесь мы проверяем вывод, но это может быть сложно без GUI
        # Вместо этого можно проверить, что функция вызывается без ошибок

#if __name__ == '__main__':
    #unittest.main()



import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from main import load_model, preprocess_image, print_predictions  # Импортируйте функции из вашего файла

class TestApp(unittest.TestCase):
    
    @patch('main.EfficientNetB0')
    def test_load_model(self, mock_EfficientNetB0):
        mock_model = MagicMock()
        mock_EfficientNetB0.return_value = mock_model
        model = load_model()
        self.assertEqual(model, mock_model)
        mock_EfficientNetB0.assert_called_once_with(weights='imagenet')
        
    def test_preprocess_image(self):
        img = Image.new('RGB', (224, 224), color=(73, 109, 137))
        x = preprocess_image(img)
        expected_shape = (1, 224, 224, 3)
        self.assertEqual(x.shape, expected_shape)
        self.assertIsInstance(x, np.ndarray)

    @patch('main.decode_predictions')
    def test_print_predictions(self, mock_decode_predictions):
        preds = np.array([[0.1, 0.2, 0.7]])
        mock_decode_predictions.return_value = [('cat', 0.1), ('dog', 0.2), ('bird', 0.7)]
        print_predictions(preds)
        mock_decode_predictions.assert_called_once_with(preds, top=3)

if __name__ == '__main__':
    unittest.main()


   
    

