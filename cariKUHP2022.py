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
    st.title("Search Engine KUHP 2022 hanya utk keperluan pendidikan by Fabian J Manoppo")
    st.write("Cari pasal dan ayat dalam KUHP 2022 berdasarkan kata kunci.")
    
    # Memuat dataset
    data = load_data()

    # Input pencarian
    keyword = st.text_input("Masukkan kata kunci pelanggaran pidana:")
    
    if keyword:
        hasil = search_kuhp(data, keyword)
        if not hasil.empty:
            st.write(f"Ditemukan {len(hasil)} pasal yang sesuai dengan kata kunci '{keyword}':")
            # Loop through results and display each pasal in a markdown format
            for index, row in hasil.iterrows():
                st.markdown(f"### {row['Pasal']}")
                st.markdown(f"{row['Isi']}")
        else:
            st.write(f"Tidak ditemukan pasal yang sesuai dengan kata kunci '{keyword}'.")

if __name__ == "__main__":
    main()
