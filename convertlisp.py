import streamlit as st

st.set_page_config(page_title="AutoLISP Generator", layout="centered")
st.title("🧱 Generator LISP Potongan Pondasi Batu Kali")

st.markdown("Masukkan dimensi pondasi di bawah ini:")

# Form Input
lebar_atas = st.text_input("Lebar pondasi atas (mm):", "600")
lebar_bawah = st.text_input("Lebar pondasi bawah (mm):", "800")
tinggi_pondasi = st.text_input("Tinggi pondasi (mm):", "700")

# Generate LISP Code
def generate_lisp(lebar_atas, lebar_bawah, tinggi_pondasi):
    return f"""
; Auto-generated LISP Potongan Pondasi
(defun c:ppbk ()
  (princ "\\nProgram Potongan Pondasi Batu Kali")
  (setq lebarpondasiatas "{lebar_atas}")
  (setq lebarpondasibawah "{lebar_bawah}")
  (setq tinggipondasibatukali "{tinggi_pondasi}")
  (princ "\\nLebar Atas: ")
  (princ lebarpondasiatas)
  (princ "\\nLebar Bawah: ")
  (princ lebarpondasibawah)
  (princ "\\nTinggi: ")
  (princ tinggipondasibatukali)
  (princ)
)
"""

# Tombol Buat File
if st.button("🎯 Buat File LISP"):
    lisp_code = generate_lisp(lebar_atas, lebar_bawah, tinggi_pondasi)
    st.download_button("⬇️ Download File LISP", lisp_code, file_name="ppbk.lsp")
