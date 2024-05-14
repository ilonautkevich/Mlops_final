# Импорт необходимых библиотек
import io
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input, decode_predictions

# Загрузка модели
# st.cache_data
def load_model():
    return EfficientNetB0(weights='imagenet')

# Функция для обработка изображения
def preprocess_image(img):
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# Функция для загрузки изображения
def load_image():
    uploaded_file = st.file_uploader(label='Выберите изображение')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

# Функция для предсказания типа изображения
def print_predictions(preds):
    classes = decode_predictions(preds, top=3)[0]
    for cl in classes:
        st.write(cl[1], cl[2])

# Создаем образец модели
model = load_model()

# Намечаем с помощью стримлит основные элементы страницы
st.title('улучшенная классификации изображений в Streamlit')
img = load_image()
result = st.button('Распознать изображение')
if result:
    x = preprocess_image(img)
    preds = model.predict(x)
    st.write('Результаты распознавания:')
    print_predictions(preds)

#Коммит 1
