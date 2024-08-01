import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Contoh data
data = {
    'kohesi': [20, 25, 30, 35, 40, 45, 50],
    'sudut_geser_dalam': [25, 30, 35, 40, 45, 50, 55],
    'berat_jenis': [18, 19, 20, 21, 22, 23, 24],
    'diameter_tiang': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
    'panjang_tiang': [10, 12, 14, 16, 18, 20, 22],
    'daya_dukung': [100, 150, 200, 250, 300, 350, 400]
}

# Mengonversi data menjadi DataFrame
df = pd.DataFrame(data)

# Memisahkan fitur dan target
X = df[['kohesi', 'sudut_geser_dalam', 'berat_jenis', 'diameter_tiang', 'panjang_tiang']]
y = df['daya_dukung']

# Membagi data menjadi set latih dan set uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model Random Forest
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Membuat aplikasi Streamlit
st.title('Prediksi Daya Dukung Fondasi Tiang Pancang, Bor dengan AI oleh Fabian J Manoppo')

# Input dari pengguna
c = st.number_input('Masukkan kohesi tanah (c) dalam kPa', min_value=0.0, value=25.0)
phi = st.number_input('Masukkan sudut geser dalam tanah (phi) dalam derajat', min_value=0.0, value=30.0)
gamma = st.number_input('Masukkan berat jenis tanah (γ) dalam kN/m³', min_value=0.0, value=18.0)
diameter = st.number_input('Masukkan diameter tiang (m)', min_value=0.0, value=0.5)
panjang_tiang = st.number_input('Masukkan panjang tiang (m)', min_value=0.0, value=10.0)

# Prediksi daya dukung menggunakan model
input_data = np.array([[c, phi, gamma, diameter, panjang_tiang]])
prediksi_beban = model.predict(input_data)[0]

# Menampilkan prediksi
st.write(f'Prediksi Daya Dukung Tiang Pancang: {prediksi_beban:.2f} kN')

# Menampilkan grafik
y_pred = model.predict(X_test)
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Daya Dukung Aktual')
ax.set_ylabel('Daya Dukung Prediksi')
ax.set_title('Prediksi vs. Aktual')
st.pyplot(fig)
