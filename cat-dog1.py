import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from pathlib import Path

# Project folder
BASE_DIR = Path(__file__).parent

st.title("CNN Image Classification")
st.write("Upload an image to classify it")

# Load saved model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        BASE_DIR / "CAT.keras"
    )

model = load_model()

# Load class names
with open(BASE_DIR / "classes.json", "r") as f:
    class_names = json.load(f)

# Upload image
file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if file is not None:

    img = Image.open(file).convert("RGB")

    st.image(img, caption="Uploaded Image")

    if st.button("Predict"):

        img = img.resize((64, 64))

        x = tf.keras.utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        prediction = model.predict(x, verbose=0)[0]

        index = np.argmax(prediction)

        st.success(
            "Predicted Category: " + class_names[index]
        )

        confidence = float(prediction[index]) * 100

        st.write(
            "Confidence:", round(confidence, 2), "%"
        )
