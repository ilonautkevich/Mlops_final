import pytest
from PIL import Image
import numpy as np
from unittest.mock import MagicMock

def test_preprocess_image():
    test_img = Image.new('RGB', (224, 224))
    x = preprocess_image(test_img)
    assert isinstance(x, np.ndarray)
    assert x.shape == (1, 224, 224, 3)


def test_load_image():
    uploaded_file = MagicMock()
    uploaded_file.getvalue.return_value = b'dummy_image_data'
    uploaded_file.content_type = 'image/jpeg'
    st.file_uploader = MagicMock(return_value=uploaded_file)
    
    img = load_image()
    assert img is not None
    assert isinstance(img, Image.Image)

def test_print_predictions(capfd):
    preds = np.array([[[0, 0, 0, 0.2, 0.5, 0.3]]])
    print_predictions(preds)
    out, _ = capfd.readouterr()
    assert '0.2' in out

def test_load_model():
    model = load_model()
    assert model is not None



