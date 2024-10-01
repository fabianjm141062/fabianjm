import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Data sentimen dan jumlah mention dari media sosial (Contoh data)
data = {
    'Candidate': ['Kandidat 1 (YSK & VM)', 'Kandidat 2 (EL & JHP)', 'Kandidat 3 (SK & DT'],
    'Sentiment': [0.7, 0.4, 0.5],  # Skor sentimen dari analisis
    'Mentions': [1200, 900, 800]    # Jumlah mention di media sosial
}

# Fitur berupa Sentimen dan Jumlah Mention
features = [[0.7, 1200], [0.4, 900], [0.5, 800]]
# Label pemenang (1 untuk menang, 0 untuk kalah)
labels = [1, 0, 0]

# Split data menjadi train dan test set (Untuk simulasi saja)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Inisialisasi Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediksi hasil test set
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit App
st.title("Prediksi Pemenang Gubernur Sulawesi Utara dengan Machine Learning AI oleh Fabian J Manoppo")
st.write("Prediksi pemenang berdasarkan data sentimen dan popularitas di media sosial.")

# Input data untuk prediksi
mention = st.number_input("Masukkan jumlah mention kandidat di media sosial", min_value=0, step=50)
sentiment = st.slider("Skor sentimen dari media sosial", -1.0, 1.0, 0.0)

# Inisialisasi model sentimen
analyzer = SentimentIntensityAnalyzer()

# Analisis sentimen teks (contoh sederhana)
text = st.text_input("Masukkan teks media sosial untuk dianalisis sentimennya:")
if text:
    sentiment_score = analyzer.polarity_scores(text)
    st.write("Skor sentimen teks:", sentiment_score['compound'])

# Prediksi pemenang berdasarkan input sentimen dan jumlah mention
if st.button("Prediksi"):
    prediksi = model.predict([[sentiment, mention]])
    hasil = "Menang" if prediksi[0] == 1 else "Kalah"
    st.write(f"Prediksi Hasil: {hasil}")

# Menampilkan akurasi model
st.write(f"Akurasi Model: {accuracy:.2f}")

# Visualisasi data kandidat
st.write("Data Sentimen dan Mention dari 3 Kandidat:")
for i, candidate in enumerate(data['Candidate']):
    st.write(f"{candidate}: Sentimen {data['Sentiment'][i]}, Mention {data['Mentions'][i]}")
