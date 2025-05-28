import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# ID file dari Google Drive
file_id = "1ncmFW6XbXAXHj_9Ort-QFA5bzXbZh-EB"
model_file = "pneumonia_model.h5"

# Download model jika belum ada
@st.cache_resource
def load_model():
    if not os.path.exists(model_file):
        with st.spinner("📥 Mengunduh model dari Google Drive..."):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", model_file, quiet=False)
    return tf.keras.models.load_model(model_file)

model = load_model()

# Fungsi preprocessing gambar
def preprocess_image(image):
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# UI Streamlit
st.title("🩺 Deteksi Radang Paru-Paru (Pneumonia) dari Citra Rontgen")
st.write("Unggah gambar X-ray dada untuk mendeteksi apakah **NORMAL** atau **RADANG PARU (PNEUMONIA)**.")

uploaded_file = st.file_uploader("📤 Unggah Gambar Rontgen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang Diupload", use_column_width=True)

    if st.button("🔍 Deteksi Sekarang"):
        with st.spinner("Menganalisis gambar..."):
            processed = preprocess_image(image)
            prediction = model.predict(processed)[0][0]
            label = "PNEUMONIA (RADANG PARU)" if prediction > 0.5 else "NORMAL"
            confidence = prediction if prediction > 0.5 else 1 - prediction
            st.success(f"**Hasil: {label}** dengan keyakinan **{confidence:.2%}**")
