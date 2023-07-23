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

model_resnet50 = models.resnet50()
# Используйте размерность [1000, 2048] для fc
model_resnet50.fc = nn.Linear(2048, 1000)
# Загрузка весов модели
model_resnet50.load_state_dict(torch.load('Weights/weight_model-2.pth'))
# Изменение размерности выходного слоя
model_resnet50.fc = nn.Linear(2048, 1)
# # Установка режима оценки (evaluation mode)
model_resnet50.eval()


def process_image_resnet50(image):
    image = image.resize((224, 224))
    image = ToTensor()(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model_resnet50(image)
        predicted_prob = torch.sigmoid(outputs)
    return predicted_prob.item()


st.sidebar.markdown(
    "# Определение вида родинок (доброкачественные/злокачественные) c помощью NN **resnet50**")


st.title("Загрузите сюда изображение родинки")
uploaded_file3 = st.file_uploader(
    'Выберите файл (jpg, jpeg, png)', type=["jpg", "jpeg", "png"])

if uploaded_file3 is not None:
    image = Image.open(uploaded_file3)
    st.image(image, caption='Загруженное изображение',
             use_column_width=True)

    predicted_prob = process_image_resnet50(image)

    if predicted_prob >= 0.5:
        st.write("Вероятность, что эта родинка относится к злокачественным",
                 predicted_prob)
    else:
        st.write("Вероятность , что эта родинка относится к доброкачественным",
                 100 - predicted_prob)
