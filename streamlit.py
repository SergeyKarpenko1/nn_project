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

# Load models
model_resnet18 = models.resnet18()
model_resnet18.fc = nn.Linear(512, 1)
model_resnet18.load_state_dict(torch.load(
    '/Users/sergey/ds_bootcamp/ds-phase-2/08-nn/nn_project/resnet18_weights.pth'))
model_resnet18.eval()

model_inceptionV3 = models.inception_v3(pretrained=True)
model_inceptionV3.eval()

model_resnet50 = models.resnet50()
# Используйте размерность [1000, 2048] для fc
model_resnet50.fc = nn.Linear(2048, 1000)
# Загрузка весов модели
model_resnet50.load_state_dict(torch.load(
    '/Users/sergey/ds_bootcamp/ds-phase-2/08-nn/nn_project/weight_model-2.pth'))
# Изменение размерности выходного слоя
model_resnet50.fc = nn.Linear(2048, 1)
# Установка режима оценки (evaluation mode)
model_resnet50.eval()

# Load class labels for the inception model
labels = json.load(open('/Users/sergey/Downloads/imagenet_class_index.json'))


def decode(class_idx):
    return labels[str(class_idx)][1]


def process_image_inception(image):
    image = image.resize((224, 224))
    image = ToTensor()(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model_inceptionV3(image)
        predicted_class = torch.argmax(outputs)
    return predicted_class.item()


def main_page():
    st.sidebar.markdown(
        "# Определение любой картинки с помощью inceptionV3_imgnet 🖼")

    st.title("Загрузите сюда любую картинку")
    uploaded_file1 = st.file_uploader(
        "Выберите изображение", type=["jpg", "jpeg", "png"])

    if uploaded_file1 is not None:
        image = Image.open(uploaded_file1)
        st.image(image, caption='Загруженное изображение',
                 use_column_width=True)

        predicted_class = process_image_inception(image)
        class_name = decode(predicted_class)
        st.write(f"Предсказанный класс: {class_name}")


def process_image_resnet(image):
    image = image.resize((224, 224))
    image = ToTensor()(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model_resnet18(image)
        predicted_prob = torch.sigmoid(outputs)
    return predicted_prob.item()


def page2():
    st.sidebar.markdown(
        "# Определение кто на картинке котик или песик с помощью NN **resnet18**")

    st.title("Загрузите сюда изображения котика или песика")
    uploaded_file = st.file_uploader(
        "Выберите изображение", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Загруженное изображение',
                 use_column_width=True)

        predicted_prob = process_image_resnet(image)

        if predicted_prob >= 0.5:
            st.write("Это песик", predicted_prob)
        else:
            st.write("Это котик", 100 - predicted_prob)


def process_image_resnet50(image):
    image = image.resize((224, 224))
    image = ToTensor()(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model_resnet50(image)
        predicted_prob = torch.sigmoid(outputs)
    return predicted_prob.item()


def page3():
    st.sidebar.markdown(
        "# Определение вида родинок (доброкачественные/злокачественные) c помощью NN **resnet50**")
    # st.markdown("# resnet50 😱")

    st.title("Загрузите сюда изображение родинки")
    uploaded_file = st.file_uploader(
        "Выберите изображение", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Загруженное изображение',
                 use_column_width=True)

        predicted_prob = process_image_resnet50(image)

        if predicted_prob >= 0.5:
            st.write("Вероятность, что эта родинка относится к злокачественным",
                     predicted_prob)
        else:
            st.write("Вероятность , что эта родинка относится к доброкачественным",
                     100 - predicted_prob)


# Mapping of page names to corresponding functions
page_names_to_funcs = {
    "NN для любой картинки": main_page,
    "NN для определения котик или песик": page2,
    "NN для определения вида родинки": page3,
}

selected_page = st.sidebar.selectbox(
    "# Выбирите NN соответствующую вашим запросам ", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
