import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

# Load model
model = load_model("models/plant_disease_model.h5")

# Class names (must match dataset folder names)
dataset_path = "D:\\AI_ML_Intership\\Plant_Disease_Detection\\dataset"
class_names = sorted(os.listdir(dataset_path))

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    class_id = np.argmax(prediction)
    confidence = np.max(prediction)

    return class_names[class_id], confidence

# Example usage
if __name__ == "__main__":
    test_image = "D:\\AI_ML_Intership\\Plant_Disease_Detection\\images.jpg" 
    label, confidence = predict_image(test_image)
    print(f"Prediction: {label} ({confidence*100:.2f}%)")