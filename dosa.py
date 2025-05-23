import streamlit as st
import pandas as pd

st.set_page_config(page_title="Cek Dosa", layout="centered")
st.title("📖 Deteksi Dosa Berdasarkan Pilihan Perilaku, Oleh Dr. Fabian J Manoppo AI Data Analyst")

# Data dosa dan ayat Alkitab
data = [
    {"Perilaku": "Membunuh", "Ayat": "Keluaran 20:13"},
    {"Perilaku": "Berzinah", "Ayat": "Markus 10:19"},
    {"Perilaku": "Iri hati", "Ayat": "Galatia 5:21"},
    {"Perilaku": "Bersaksi dusta", "Ayat": "Keluaran 20:16"},
    {"Perilaku": "Menyimpan kebencian", "Ayat": "1 Yohanes 3:15"},
    {"Perilaku": "Tidak mengampuni", "Ayat": "Matius 6:15"},
    {"Perilaku": "Mencuri", "Ayat": "Keluaran 20:15"},
    {"Perilaku": "Menyembah berhala", "Ayat": "Keluaran 20:3-5"},
]

df = pd.DataFrame(data)

st.markdown("### Pilih 'Yes' jika Anda pernah melakukannya:")

# Input pengguna
dosa_count = 0
for i, row in df.iterrows():
    pilihan = st.radio(
        f"{row['Perilaku']} ({row['Ayat']})",
        ["No", "Yes"],
        key=row['Perilaku'],
        horizontal=True
    )
    if pilihan == "Yes":
        dosa_count += 1

# Output jumlah dosa
st.markdown(f"### Jumlah dosa yang dipilih: **{dosa_count}**")

# Output pesan sesuai jumlah dosa
if dosa_count > 0:
    st.markdown("<h3 style='color:red'>🛐 Bertobatlah Karena Kerajaan Surga Sudah Dekat</h3>", unsafe_allow_html=True)
else:
    st.markdown("<h3 style='color:green'>✨ Berbahagialah, upahmu besar di sorga</h3>", unsafe_allow_html=True)
