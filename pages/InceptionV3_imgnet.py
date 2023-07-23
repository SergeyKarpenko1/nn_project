import json

from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import ToTensor
import streamlit as st

import ssl
import urllib.request


# def InceptionV3_imgnet_page(image):
model_inceptionV3 = models.inception_v3(pretrained=True)
model_inceptionV3.eval()
# return model_inceptionV3(image)


# Load class labels for the inception model
labels = json.load(open('imagenet_class_index.json'))


def decode(x): return labels[str(x)][1]

# def decode(class_idx):
#     return labels[str(class_idx)][1]


def process_image_inception(image):
    image = image.resize((224, 224))
    image = ToTensor()(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model_inceptionV3(image)
        predicted_class = torch.argmax(outputs)
    return predicted_class.item()


st.sidebar.markdown(
    "# Определение любой картинки с помощью inceptionV3_imgnet ")

st.title("Загрузите сюда любую картинку")
uploaded_image = st.file_uploader(
    'Выберите файл (jpg, jpeg, png)', type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Загруженное изображение', use_column_width=True)

    predicted_class = process_image_inception(image)
    class_name = decode(predicted_class)
    st.write(f"Предсказанный класс: {class_name}")
