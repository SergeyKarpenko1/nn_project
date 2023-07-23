import streamlit as st


import json

from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.transforms import ToTensor
import streamlit as st

import ssl
import urllib.request

# st.set_page_config(page_title="Cat and Dogs", page_icon="🐶")

# Load models
model_resnet18 = models.resnet18()
model_resnet18.fc = nn.Linear(512, 1)
model_resnet18.load_state_dict(torch.load('Weights/resnet18_weights.pth'))
model_resnet18.eval()


def process_image_resnet(image):
    image = image.resize((224, 224))
    image = ToTensor()(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model_resnet18(image)
        predicted_prob = torch.sigmoid(outputs)
    return predicted_prob.item()


# def page2():
st.sidebar.markdown(
    "# Определение кто на картинке котик или песик с помощью NN **resnet18**")

st.title("Загрузите сюда изображения котика или песика")
uploaded_file = st.file_uploader(
    'Выберите файл (jpg, jpeg, png)', type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Загруженное изображение',
             use_column_width=True)

    predicted_prob = process_image_resnet(image)

    if predicted_prob >= 0.5:
        st.write("Это песик", predicted_prob)
    else:
        st.write("Это котик", 100 - predicted_prob)
