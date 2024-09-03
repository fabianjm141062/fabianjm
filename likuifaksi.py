import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk menghitung CSR (Cyclic Stress Ratio)
def calculate_csr(PGA, sigma_v, sigma_v_prime, rd):
    g = 9.81  # percepatan gravitasi m/s^2
    CSR = 0.65 * (PGA / g) * (sigma_v / sigma_v_prime) * rd
    return CSR

# Fungsi untuk menghitung CRR berdasarkan nilai SPT (N1,60)
def calculate_crr(N1_60):
    # Koreksi untuk nilai CRR berdasarkan N1,60 (nilai perkiraan)
    if N1_60 < 15:
        CRR = 0.833 * (N1_60 / 14) + 0.05
    elif N1_60 < 30:
        CRR = 0.833 * (N1_60 / 14) + 0.05
    else:
        CRR = 1.0  # Asumsi konservatif
    return CRR

# Stress reduction factor (rd) untuk kedalaman
def stress_reduction_factor(depth):
    if depth < 9.15:
        rd = 1.0 - 0.00765 * depth
    else:
        rd = 1.174 - 0.0267 * depth
    return rd

# Fungsi untuk menghitung faktor keamanan (FS)
def calculate_fs(PGA, sigma_v, sigma_v_prime, N1_60, depth):
    rd = stress_reduction_factor(depth)
    CSR = calculate_csr(PGA, sigma_v, sigma_v_prime, rd)
    CRR = calculate_crr(N1_60)
    FS = CRR / CSR
    return FS

# Streamlit UI
st.title("Prediksi Likuifaksi Menggunakan Metode Seed & Idriss 1971 oleh Fabian J Manoppo")

# Input data lapisan tanah
st.header("Data Lapisan Tanah")

gamma1 = st.number_input("Berat satuan tanah untuk lapisan 1 (kN/m³)", value=18.0)
h1 = st.number_input("Ketebalan lapisan 1 (meter)", value=2.0)
N1_60_1 = st.number_input("Nilai SPT (N1,60) untuk lapisan 1", value=10.0)

gamma2 = st.number_input("Berat satuan tanah untuk lapisan 2 (kN/m³)", value=19.0)
h2 = st.number_input("Ketebalan lapisan 2 (meter)", value=3.0)
N1_60_2 = st.number_input("Nilai SPT (N1,60) untuk lapisan 2", value=15.0)

gamma3 = st.number_input("Berat satuan tanah untuk lapisan 3 (kN/m³)", value=20.0)
h3 = st.number_input("Ketebalan lapisan 3 (meter)", value=4.0)
N1_60_3 = st.number_input("Nilai SPT (N1,60) untuk lapisan 3", value=20.0)

# Input data lain
gamma_w = 10  # Berat satuan air, asumsi 10 kN/m³
GWT = st.number_input("Kedalaman muka air tanah (GWT) (meter)", value=2.0)
kedalaman = st.number_input("Kedalaman untuk perhitungan (meter)", value=5.0)
SR = st.number_input("Besaran gempa (SR) [5-9 SR]", value=6.0)

# Konversi SR ke PGA menggunakan pendekatan sederhana (misalnya: hubungan linear)
PGA = 0.1 * SR  # Contoh sederhana: 0.1 g untuk setiap unit SR

# Tentukan N1_60 berdasarkan kedalaman
if kedalaman <= h1:
    N1_60 = N1_60_1
    sigma_v = gamma1 * kedalaman
elif kedalaman <= (h1 + h2):
    N1_60 = N1_60_2
    sigma_v = (gamma1 * h1) + (gamma2 * (kedalaman - h1))
elif kedalaman <= (h1 + h2 + h3):
    N1_60 = N1_60_3
    sigma_v = (gamma1 * h1) + (gamma2 * h2) + (gamma3 * (kedalaman - h1 - h2))
else:
    st.error("Kedalaman melebihi total lapisan tanah yang diberikan.")
    st.stop()

# Hitung tekanan air pori
if kedalaman > GWT:
    u = gamma_w * (kedalaman - GWT)
else:
    u = 0  # Di atas muka air tanah, tidak ada tekanan air pori

# Hitung tegangan vertikal efektif
sigma_v_prime = sigma_v - u

# Tampilkan hasil perhitungan tegangan
st.write(f"Tegangan Vertikal Total (σ_v): {sigma_v:.2f} kPa")
st.write(f"Tegangan Vertikal Efektif (σ'_v): {sigma_v_prime:.2f} kPa")

# Hitung faktor keamanan (FS)
FS = calculate_fs(PGA, sigma_v, sigma_v_prime, N1_60, kedalaman)
st.write(f'Faktor Keamanan (FS): {FS:.2f}')

# Prediksi likuifaksi
if FS < 1:
    st.error("Likuifaksi diprediksi terjadi.")
else:
    st.success("Likuifaksi tidak diprediksi terjadi.")

# Visualisasi FS terhadap kedalaman
depths = np.linspace(1, kedalaman, 100)  # Membuat range kedalaman hingga nilai input
FS_values = [calculate_fs(PGA, sigma_v, sigma_v_prime, N1_60, d) for d in depths]

plt.figure(figsize=(10, 6))
plt.plot(FS_values, depths, label='Faktor Keamanan (FS)')
plt.axvline(x=1, color='r', linestyle='--', label='Batas Likuifaksi (FS=1)')
plt.xlabel('Faktor Keamanan (FS)')
plt.ylabel('Kedalaman (meter)')
plt.title('Prediksi Likuifaksi Berdasarkan Kedalaman Teori Sheed & Idris 1971 oleh Fabian J Manoppo')
plt.gca().invert_yaxis()  # Membalikkan sumbu Y untuk menunjukkan kedalaman dari atas ke bawah
plt.grid(True)
plt.legend()
st.pyplot(plt)
