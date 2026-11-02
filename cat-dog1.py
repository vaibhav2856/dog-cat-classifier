import streamlit as st
import tensorflow as tf
import json
import numpy as np
from PIL import Image

# Page Title
st.title("CNN Image Classification")
st.write("Upload an image to classify it")

# Load Images
data = tf.keras.utils.image_dataset_from_directory(
    r"C:\vaibhav\dog-cat-class\train",
    image_size=(64, 64),
    batch_size=2
)

# CNN Model
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(16, (3,3), activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(5, activation="softmax")
])

# Compile Model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train Model
if st.button("Train Model"):
    with st.spinner("Training model..."):
        model.fit(data, epochs=50)
        model.save("CAT.keras")

        with open("classes.json", "w") as f:
            json.dump(data.class_names, f)

    st.success("Model trained successfully!")

# Upload Image
file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

if file is not None:

    img = Image.open(file).convert("RGB")

    st.image(img, caption="Uploaded Image")

    if st.button("Predict Image"):

        # Resize Image
        img = img.resize((64,64))

        # Convert Image to Array
        x = tf.keras.utils.img_to_array(img)

        # Expand Dimensions
        x = tf.expand_dims(x, 0)

        # Load Saved Model
        m = tf.keras.models.load_model("CAT.keras")

        # Load Class Names
        with open("classes.json", "r") as f:
            class_names = json.load(f)

        # Prediction
        p = m.predict(x)

        # Display Result
        st.success("Predicted Category: " + class_names[np.argmax(p)])

        st.write("Prediction:", p)
        st.write("Accuracy:", round(float(np.max(p))*100, 2), "%")