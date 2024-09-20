import streamlit as st
import pandas as pd

# Fungsi untuk memuat dataset yang telah dibersihkan
@st.cache
def load_data():
    return pd.read_csv('dataset_KUHP_2022.csv')

# Fungsi untuk mencari pasal berdasarkan kata kunci
def search_kuhp(data, keyword):
    hasil = data[data['Isi'].str.contains(keyword, case=False, na=False)]
    return hasil if not hasil.empty else pd.DataFrame(columns=data.columns)

# Program utama Streamlit
def main():
    st.title("Search Engine KUHP 2022 by Fabian J Manoppo")
    st.write("Cari pasal dan ayat dalam KUHP 2022 berdasarkan kata kunci.")
    
    # Memuat dataset
    data = load_data()

    # Input pencarian
    keyword = st.text_input("Masukkan kata kunci pelanggaran pidana:")
    
    if keyword:
        hasil = search_kuhp(data, keyword)
        if not hasil.empty:
            st.write(f"Ditemukan {len(hasil)} pasal yang sesuai dengan kata kunci '{keyword}':")
            st.dataframe(hasil[['Pasal', 'Isi']])
        else:
            st.write(f"Tidak ditemukan pasal yang sesuai dengan kata kunci '{keyword}'.")

if __name__ == "__main__":
    main()
