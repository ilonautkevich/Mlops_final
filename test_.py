import numpy as np
from PIL import Image
from keras.preprocessing import image
from preprocess import preprocess_image

# Test case 1: Test image resize
img = Image.open("test1.jpg")
result = preprocess_image(img)
assert result.shape == (1, 224, 224, 3)

# Test case 2: Test image to array conversion
img = Image.open("test2.jpg")
result = preprocess_image(img)
assert isinstance(result, np.ndarray)

# Test case 3: Test image preprocessing
img = Image.open("test3.jpg")
result = preprocess_image(img)
assert np.all(np.isnan(result) == False)

print("All tests passed successfully!")