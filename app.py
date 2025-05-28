import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model hanya sekali
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("pneumonia_model.h5")

model = load_model()

# Preprocessing image
def preprocess_image(image):
    image = image.resize((150, 150))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Streamlit UI
st.title("🩺 Deteksi Pneumonia dari Citra Rontgen Paru-paru, oleh Dr. Fabian J Manoppo AI Data Analyst")
st.markdown("Upload gambar X-ray dada untuk mendeteksi apakah **NORMAL** atau **PNEUMONIA**.")

uploaded_file = st.file_uploader("📤 Upload Gambar Rontgen", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar Diupload", use_column_width=True)

    if st.button("🔍 Analisis Sekarang"):
        with st.spinner("Sedang menganalisis..."):
            processed = preprocess_image(image)
            prediction = model.predict(processed)[0][0]
            label = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
            confidence = prediction if prediction > 0.5 else 1 - prediction

            st.subheader("🧠 Hasil Prediksi:")
            st.success(f"**{label}** dengan keyakinan **{confidence:.2%}**")
