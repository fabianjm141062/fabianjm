import streamlit as st
import pandas as pd
from gtts import gTTS
import os

st.title("Deteksi Dosa Berdasarkan Perilaku, Oleh D. Fabian J Manoppo AI Data Analyst")

# Dataset
data = {
    "Perilaku": ["Membunuh", "Memberi sedekah", "Berzinah", "Iri hati", "Mengampuni"],
    "Dosa (Label)": ["Ya", "Tidak", "Ya", "Ya", "Tidak"],
    "Sumber": ["10 Perintah Allah", "Matius 6", "Markus 10", "Galatia 5", "Matius 18"]
}
df = pd.DataFrame(data)
st.dataframe(df)

# Logika dosa
if "Ya" in df["Dosa (Label)"].values:
    pesan = "Bertobatlah Karena Kerajaan Surga Sudah Dekat"
    warna = "red"
else:
    pesan = "Berbahagialah, upahmu besar di sorga"
    warna = "green"

st.markdown(f"<h3 style='color:{warna}'>{pesan}</h3>", unsafe_allow_html=True)

# Buat suara dengan gTTS
tts = gTTS(pesan)
tts.save("pesan.mp3")

# Tampilkan audio player
audio_file = open("pesan.mp3", "rb")
audio_bytes = audio_file.read()
st.audio(audio_bytes, format="audio/mp3")
