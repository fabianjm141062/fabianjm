import streamlit as st
import pandas as pd

st.set_page_config(page_title="Deteksi Dosa", layout="centered")

st.title("📖 Deteksi Dosa Berdasarkan Perilaku, oleh Dr.Fabian J Manoppo AI Data Analyst")

# Dataset
data = {
    "Perilaku": ["Membunuh", "Memberi sedekah", "Berzinah", "Iri hati", "Mengampuni"],
    "Dosa (Label)": ["Ya", "Tidak", "Ya", "Ya", "Tidak"],
    "Sumber": ["10 Perintah Allah", "Matius 6", "Markus 10", "Galatia 5", "Matius 18"]
}
df = pd.DataFrame(data)

st.subheader("📋 Daftar Perilaku dan Label Dosa")
st.dataframe(df, use_container_width=True)

# Deteksi dosa
if "Ya" in df["Dosa (Label)"].values:
    pesan = "🛐 **Bertobatlah Karena Kerajaan Surga Sudah Dekat**"
    warna = "red"
else:
    pesan = "✨ **Berbahagialah, upahmu besar di sorga**"
    warna = "green"

# Tampilkan pesan
st.markdown(f"<h3 style='color:{warna}'>{pesan}</h3>", unsafe_allow_html=True)
