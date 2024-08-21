import streamlit as st
import numpy as np
import tensorflow as tf

# Memuat model
model = tf.keras.models.load_model('structure_health_model.h5')

# Fungsi prediksi dengan aturan tambahan
def predict_health_status(vibration, temperature, humidity, displacement):
    if vibration <= 0.03 and temperature <= 30 and humidity <= 45 and displacement <= 0.002:
        return 0  # Sehat
    elif 0.03 < vibration <= 0.07 or 30 < temperature <= 40 or 45 < humidity <= 55 or 0.002 < displacement <= 0.005:
        return 1  # Perlu Tinjau
    elif vibration > 0.07 or temperature > 40 or humidity > 55 or displacement > 0.005:
        return 2  # Rusak
    else:
        X_new = np.array([[vibration, temperature, humidity, displacement]])
        prediction = model.predict(X_new)
        return np.argmax(prediction, axis=1)[0]

# Judul aplikasi
st.title('Structure Health Monitoring - 20 Floor Building')

# Input data dari pengguna
vibration = st.number_input('Vibration (0-1)', min_value=0.0, max_value=1.0, value=0.05)
temperature = st.number_input('Temperature (°C)', min_value=0, max_value=100, value=30)
humidity = st.number_input('Humidity (%)', min_value=0, max_value=100, value=50)
displacement = st.number_input('Displacement (0-0.01)', min_value=0.0, max_value=0.01, value=0.002)

# Prediksi status kesehatan
if st.button('Predict Health Status'):
    status = predict_health_status(vibration, temperature, humidity, displacement)
    status_dict = {0: 'Sehat', 1: 'Perlu Tinjau', 2: 'Rusak'}
    st.write(f"Prediksi Status Kesehatan: {status_dict[status]}")

