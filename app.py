import streamlit as st
import numpy as np
from PIL import Image
import cv2
from utils import set_background
import tensorflow as tf
import tensorflow_hub as hub

set_background("./imgs/background.png")

MODEL_PATH = ""

header = st.container()
body = st.container()

model = tf.keras.models.load_model(
            (MODEL_PATH),
            custom_objects={'KerasLayer':hub.KerasLayer}, compile=False
        )

threshold = 0.15

def model_prediction(img):
    patient_result = ""

    img_resized = cv2.resize(img, (256, 256), interpolation = cv2.INTER_AREA)
    img_format = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    predict_img = np.expand_dims(img_format, axis = 0)

    predict_img = np.vstack([predict_img])

    result = model.predict(predict_img)

    print(result)

    arg_max_result = np.argmax(result)
    per = result[0][arg_max_result] * 100

    if arg_max_result == 0 :
        patient_result = f"The Patient has Covid | Percentage: {int(per)} %"
    elif arg_max_result == 1 :
        patient_result = f"The Patient has a Normal X-Ray | Percentage: {int(per)} %"
    elif arg_max_result == 2 :
        patient_result = f"The Patient has Viral Pneumonia | Percentage: {int(per)} %"  

    return img_format, patient_result

    
with header :
    _, col1, _ = st.columns([0.15,1,0.1])
    col1.title("💉 Covid and Pneumonia Recognition with X-Ray 🦠")

    _, col4, _ = st.columns([0.2,1,0.2])
    col4.subheader("Computer Vision Project with TensorFlow and CNN 🧪")

    _, col5, _ = st.columns([0.2,1,0.1])
    col5.image("./imgs/0100.jpeg", width=450)

    st.write("The Multi Classification Model, was trained with over 300 X-Ray Images, 100 Images for each class, Covid, Pneumonia and Normal X-Ray. Also using TensorFlow and with the CNN Architecture.")

with body :
    _, col1, _ = st.columns([0.2,1,0.2])
    col1.subheader("Check It-out the Covid and Pneumonia Recognition Model 🔎!")

    img = st.file_uploader("Upload a X-Ray Image:", type=["png", "jpg", "jpeg"])

    _, col2, _ = st.columns([0.3,1,0.2])

    _, col5, _ = st.columns([0.8,1,0.2])

    
    if img is not None:
        image = np.array(Image.open(img))    
        col2.image(image, width=400)

        if col5.button("Analyze X-Ray"):
            xray_img, patient_result = model_prediction(image)

            _, col3, _ = st.columns([0.7,1,0.2])
            col3.header("Results ✅:")
                    
            _, col4, _ = st.columns([0.1,1,0.1])
            col4.success(patient_result)




