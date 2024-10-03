import streamlit as st
import random
import matplotlib.pyplot as plt

# -------------------------- Data Kandidat --------------------------------
candidates = {
    'Yulius-Victor': 0,
    'Lasut-Hanny': 0,
    'Steven-Alfred': 0
}

# Jumlah responden yang ikut dalam survei (misalnya 100 orang)
num_respondents = st.slider("Jumlah Responden dalam Survei:", 10, 500, 100)

# Fungsi untuk melakukan survei menggunakan random sampling
def conduct_survey(num_respondents):
    survey_results = {'Yulius-Victor': 0, 'Lasut-Hanny': 0, 'Steven-Alfred': 0}
    for _ in range(num_respondents):
        chosen_candidate = random.choice(list(survey_results.keys()))
        survey_results[chosen_candidate] += 1
    return survey_results

# Fungsi untuk menampilkan hasil survei dalam bentuk grafik
def display_survey_results(survey_results):
    candidate_names = list(survey_results.keys())
    votes = list(survey_results.values())

    # Membuat grafik batang
    fig, ax = plt.subplots()
    ax.bar(candidate_names, votes, color=['blue', 'green', 'orange'])
    ax.set_title('Prediksi dengan AI Perolehan Suara Pilkada Sulawesi Utara tahun 2024')
    ax.set_xlabel('Pasangan Calon')
    ax.set_ylabel('Jumlah Suara')

    st.pyplot(fig)

# -------------------------- Main Program --------------------------------
st.title("Prediksi dengan AI Perolehan Suara Gubernur Sulawesi Utara tahun 2024")

# Lakukan survei
survey_results = conduct_survey(num_respondents)

# Tampilkan hasil survei dalam bentuk grafik
display_survey_results(survey_results)
