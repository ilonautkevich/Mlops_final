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
