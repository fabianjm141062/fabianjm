import streamlit as st
import pytesseract
from PIL import Image
from docx import Document
import io

# Fungsi untuk melakukan OCR dari gambar
def extract_text_from_image(image):
    text = pytesseract.image_to_string(image)
    return text

# Fungsi untuk membuat file Word dari teks yang diekstrak
def create_word_from_text(extracted_text):
    doc = Document()
    doc.add_paragraph(extracted_text)
    output = io.BytesIO()
    doc.save(output)
    output.seek(0)
    return output

# Streamlit Antarmuka
st.title("Konversi Gambar ke Word")

# Instruksi untuk mengunggah gambar
uploaded_image = st.file_uploader("Unggah gambar di sini (format PNG, JPG, JPEG)", type=['png', 'jpg', 'jpeg'])

# Jika gambar diunggah, mulai proses
if uploaded_image is not None:
    # Membaca gambar menggunakan PIL
    image = Image.open(uploaded_image)
    
    # Tampilkan gambar yang diunggah
    st.image(image, caption='Gambar yang diunggah', use_column_width=True)
    
    # Ekstraksi teks menggunakan OCR
    with st.spinner('Ekstraksi teks dari gambar...'):
        extracted_text = extract_text_from_image(image)
    
    # Tampilkan teks yang diekstrak
    st.subheader("Teks yang diekstrak:")
    st.text(extracted_text)

    # Jika teks berhasil diekstrak, tawarkan opsi untuk mengunduh file Word
    if extracted_text:
        st.success("Teks berhasil diekstrak! Unduh file Word di bawah ini.")
        
        # Buat file Word
        word_file = create_word_from_text(extracted_text)
        
        # Tombol untuk mengunduh file Word
        st.download_button(
            label="Unduh sebagai Word",
            data=word_file,
            file_name="hasil_ekstraksi_teks.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
