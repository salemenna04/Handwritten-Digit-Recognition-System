
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

model = tf.keras.models.load_model("mnist_cnn.h5")

st.title("Handwritten Digit Recognition (CNN + MNIST)")
st.write("Upload your own handwritten digit image")

uploaded_file = st.file_uploader("Choose an image...", type=["png","jpg","jpeg"])

def preprocess(image):
    image = ImageOps.grayscale(image)
    image = image.resize((28,28))
    image = np.array(image)

    image = 255 - image

    image = (image > 30) * image

    image = image / 255.0
    image = image.reshape(1,28,28,1)

    return imageif uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    processed = preprocess_image(img)
    prediction = model.predict(processed)
    digit = np.argmax(prediction)

    st.success(f"Predicted Digit: {digit}")
