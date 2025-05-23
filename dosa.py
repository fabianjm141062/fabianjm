import streamlit as st

st.set_page_config(page_title="Cek Dosa", layout="centered")

st.title("📖 Deteksi Dosa: Pilih Perilaku Anda, oleh Dr.Fabian J Manoppo AI Data Analyst")

# Daftar perilaku
perilaku_list = [
    "Membunuh",
    "Berzinah",
    "Iri hati",
    "Bersaksi dusta",
    "Menyimpan kebencian",
    "Tidak mengampuni",
    "Mencuri",
    "Menyembah berhala"
]

st.markdown("Silakan pilih **Yes** jika Anda pernah melakukannya:")

# Input user
dosa_respon = {}
for perilaku in perilaku_list:
    dosa_respon[perilaku] = st.radio(perilaku, ["No", "Yes"], horizontal=True)

# Evaluasi hasil
if "Yes" in dosa_respon.values():
    st.markdown("<h3 style='color:red'>🛐 Bertobatlah Karena Kerajaan Surga Sudah Dekat</h3>", unsafe_allow_html=True)
else:
    st.markdown("<h3 style='color:green'>✨ Berbahagialah, upahmu besar di sorga</h3>", unsafe_allow_html=True)
