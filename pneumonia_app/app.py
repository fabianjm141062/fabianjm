import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("pneumonia_model.h5")
    return model

model = load_model()

# Preprocessing function
def preprocess_image(image):
    image = image.resize((150, 150))  # Adjust based on model input
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# App Interface
st.title("Deteksi Pneumonia dari Rontgen Paru")
st.write("Upload gambar rontgen dada (chest X-ray) untuk analisis Pneumonia.")

uploaded_file = st.file_uploader("Upload Gambar Rontgen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Gambar Rontgen Diupload", use_column_width=True)

    if st.button("Analisis Gambar"):
        input_image = preprocess_image(image)
        prediction = model.predict(input_image)
        
        result = "PNEUMONIA" if prediction[0][0] > 0.5 else "NORMAL"
        confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

        st.subheader("Hasil Diagnosa:")
        st.write(f"**{result}** (Confidence: {confidence:.2%})")
