import sys
import os
import cv2
import numpy as np
from PIL import Image
import streamlit as st
from keras.models import load_model

model = load_model('BrainTumor20EpochsCategorical10.h5')
st.write('Model loaded.')

IMG_WIDTH, IMG_HEIGHT = 64, 64

def get_class_name(class_no):
    return "No Brain Tumor" if class_no == 0 else "Yes Brain Tumor"

def preprocess_image(img):
    try:
        img = Image.open(img).convert('RGB')
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
        img = np.array(img)
        img = img / 255.0
        input_img = np.expand_dims(img, axis=0)
        return input_img
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

def get_result(img):
    input_img = preprocess_image(img)
    if input_img is not None:
        result = model.predict(input_img)
        predicted_class = np.argmax(result, axis=1)
        return predicted_class
    else:
        return None

st.title("Brain Tumor Classification")
st.write("Upload an MRI image to predict if there is a brain tumor.")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    if st.button("Predict"):
        value = get_result(uploaded_file)
        if value is not None:
            result = get_class_name(value[0])
            st.success(f"Prediction: {result}")
        else:
            st.error("Error processing image.")
