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

# Отключение проверки SSL-сертификата
ssl._create_default_https_context = ssl._create_unverified_context


st.set_page_config(
    page_title='Проект по Neural Network',
    layout='wide'
)
# st.sidebar.header("Home page")
c1, c2 = st.columns(2)
c2.image('neural_img.png')
c1.markdown("""
# Проект по Neural Network
Cостоит из 3 частей:
### 1.Определения кто на картинке собака или кот с помощью **NN resnet18.**
### 2.Определения что изображено на картинке с помощью **NN inceptionV3.**
### 3.Определение вида родинок c помощью **NN resnet50.**
""")
