# file: dosa_checker.py

import streamlit as st
import pandas as pd
import pyttsx3

# Inisialisasi engine suara
engine = pyttsx3.init()

# Judul aplikasi
st.title("Deteksi Dosa Berdasarkan Perilaku")

# Contoh dataset
data = {
    "Perilaku": ["Membunuh", "Memberi sedekah", "Berzinah", "Iri hati", "Mengampuni"],
    "Dosa (Label)": ["Ya", "Tidak", "Ya", "Ya", "Tidak"],
    "Sumber": ["10 Perintah Allah", "Matius 6", "Markus 10", "Galatia 5", "Matius 18"]
}

df = pd.DataFrame(data)
st.dataframe(df)

# Cek apakah ada dosa
if "Ya" in df["Dosa (Label)"].values:
    pesan = "Bertobatlah Karena Kerajaan Surga Sudah Dekat"
    warna = "red"
else:
    pesan = "Berbahagialah Upahmu Besar Disorga"
    warna = "green"

# Tampilkan hasil
st.markdown(f"<h2 style='color:{warna}'>{pesan}</h2>", unsafe_allow_html=True)

# Tombol untuk mengaktifkan suara
if st.button("🔊 Putar Suara"):
    engine.say(pesan)
    engine.runAndWait()
