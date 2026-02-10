import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

IMG_SIZE = 224

try:
    model = tf.keras.models.load_model("cats_vs_dogs_image_classifier_model.keras")
    st.success("Image Classifier Model loaded successfully!")
except:
    st.success("Image Classifier Model not found. Please make sure the model is in the same foler as Cats vs. Dogs Image Classifier.py")

st.title("Cats vs. Dogs Image Classifier")
st.write("Upload an image of cat or dog to predict if the image is of cat or dog: ")

uploaded_image = st.file_uploader("Upload an image of cat or dog: ", type = ["png", "jpg", "jpeg"])

if (uploaded_image is not None):
    image = Image.open(uploaded_image).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    st.image(image, caption = "Uploaded Image", width = IMG_SIZE)

    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis = 0)

    prediction = model.predict(image_array)[0][0]

    if (prediction >= 0.5):
        st.success(f"Prediction: Dog (Confidence: {prediction:.2f})")
    else:
        st.success(f"Prediction: Cat (Confidence: {(1 - prediction):.2f})")