import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Langkah 1: Data Dummy dan Analisis Sentimen
analyzer = SentimentIntensityAnalyzer()

# Data contoh (simulasi data dari media sosial)
data = [
    {"candidate": "Kandidat A", "text": "Saya sangat mendukung Kandidat A!", "mentions": 1500},
    {"candidate": "Kandidat B", "text": "Kandidat B tidak memiliki visi yang jelas.", "mentions": 1000},
    {"candidate": "Kandidat C", "text": "Kandidat C bisa menjadi pilihan yang lebih baik.", "mentions": 500},
    {"candidate": "Kandidat D", "text": "Kandidat D memiliki ide-ide bagus!", "mentions": 1300},
    {"candidate": "Kandidat E", "text": "Saya tidak yakin dengan Kandidat E.", "mentions": 800}
]

# Fungsi untuk mengukur skor sentimen
def get_sentiment_score(text):
    return analyzer.polarity_scores(text)['compound']

# Tambahkan skor sentimen ke data
for d in data:
    d['sentiment'] = get_sentiment_score(d['text'])

# Tampilkan data dengan skor sentimen
st.write("Data Sentimen dari Media Sosial:")
st.write(data)

# Langkah 2: Pelatihan Model Prediksi
# Membuat dataset fitur (sentimen dan mention) dan label (prediksi kemenangan: 1 menang, 0 kalah)
features = [[d['sentiment'], d['mentions']] for d in data]
labels = [1, 0, 0, 1, 0]  # Dummy labels untuk contoh (1 menang, 0 kalah)

# Split data untuk pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Inisialisasi model Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediksi hasil
y_pred = model.predict(X_test)

# Evaluasi akurasi
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Akurasi model: {accuracy}")

# Langkah 3: Prediksi dengan Input dari Pengguna
st.title("Prediksi Pemenang Gubernur Sulawesi Utara")
st.write("Masukkan data kandidat untuk melihat prediksi hasil")

# Masukkan jumlah mention dan sentimen dari pengguna
mention = st.number_input("Jumlah Mention", min_value=0, value=1000)
text = st.text_input("Masukkan teks dari media sosial")

# Hitung skor sentimen dari input pengguna
sentiment = get_sentiment_score(text)

# Lakukan prediksi
if st.button("Prediksi"):
    prediksi = model.predict([[sentiment, mention]])
    st.write(f"Prediksi: {'Menang' if prediksi == 1 else 'Kalah'}")

# Langkah 4: Visualisasi Hasil
sentiment_scores = [d['sentiment'] for d in data]
candidates = [d['candidate'] for d in data]

plt.bar(candidates, sentiment_scores)
plt.xlabel('Kandidat')
plt.ylabel('Skor Sentimen')
plt.title('Analisis Sentimen Media Sosial Kandidat Gubernur')
st.pyplot(plt)
