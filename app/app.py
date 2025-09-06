import streamlit as st
import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("models/plant_disease_model.h5")

# Class names
dataset_path = "D:\\AI_ML_Intership\\Plant_Disease_Detection\\dataset"
class_names = sorted(os.listdir(dataset_path))

st.title("Plant Disease Detection")
st.write("Upload a leaf image and the model will predict the disease.")

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, channels="BGR", caption="Uploaded Image", use_column_width=True)

    # Preprocess
    resized = cv2.resize(img, (128, 128))
    normalized = resized / 255.0
    expanded = np.expand_dims(normalized, axis=0)

    # Predict
    prediction = model.predict(expanded)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)

    st.write(f"  Prediction: *{class_names[class_id]}*")
    st.write(f"   Confidence: *{confidence*100:.2f}%*")