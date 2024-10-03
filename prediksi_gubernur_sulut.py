import tweepy
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import streamlit as st

# Twitter API v2 credentials - masukkan Bearer Token kamu
bearer_token = 'npx create-react-app real-time-tweet-streamer'

# Inisialisasi Tweepy Client untuk Twitter API v2
client = tweepy.Client(bearer_token=bearer_token)

# Inisialisasi VADER Sentiment Analyzer untuk analisis sentimen
analyzer = SentimentIntensityAnalyzer()

# Fungsi untuk mendapatkan tweet dan melakukan analisis sentimen
def get_twitter_data_real_time(keyword, count=10):
    query = f"{keyword} -is:retweet lang:id"
    tweets = client.search_recent_tweets(query=query, max_results=count, tweet_fields=['public_metrics', 'created_at'])
    
    tweet_list = []
    if tweets.data:  # Memastikan data ada
        for tweet in tweets.data:
            sentiment = analyzer.polarity_scores(tweet.text)['compound']
            sentiment_label = "Positif" if sentiment >= 0.05 else "Negatif" if sentiment <= -0.05 else "Netral"
            tweet_list.append([tweet.text, tweet.public_metrics['like_count'], tweet.public_metrics['retweet_count'], sentiment_label, tweet.created_at])
    
    return pd.DataFrame(tweet_list, columns=['Tweet', 'Likes', 'Retweets', 'Sentiment', 'Timestamp'])

# Fungsi untuk menampilkan hasil analisis dalam Streamlit
def display_sentiment_analysis(df, candidate_name):
    st.subheader(f"Analisis Sentimen untuk {candidate_name}")
    
    # Hitung persentase sentimen
    sentiment_counts = df['Sentiment'].value_counts(normalize=True) * 100
    st.bar_chart(sentiment_counts)
    
    # Tampilkan tweet dengan jumlah like tertinggi
    st.subheader(f"Tweet dengan Like Tertinggi untuk {candidate_name}")
    most_liked = df[df['Likes'] == df['Likes'].max()]
    st.write(most_liked[['Tweet', 'Likes', 'Retweets', 'Sentiment', 'Timestamp']])

# Fungsi utama untuk Streamlit
def main():
    st.title("Real-time Analisis Sentimen & Popularitas Kandidat Gubernur Sulawesi Utara")
    
    st.write("Data ini diambil secara real-time dari Twitter menggunakan API Twitter v2.")
    
    # Pilihan jumlah tweet yang akan diambil
    count = st.slider("Jumlah Tweet yang Diambil:", min_value=10, max_value=100, value=50)
    
    # Ambil data real-time untuk masing-masing kandidat
    df_yulius = get_twitter_data_real_time('Yulius Victor', count)
    df_lasut = get_twitter_data_real_time('Lasut Hanny', count)
    df_steven = get_twitter_data_real_time('Steven Alfred', count)
    
    # Tampilkan data sentimen untuk masing-masing kandidat
    display_sentiment_analysis(df_yulius, 'Yulius Victor')
    display_sentiment_analysis(df_lasut, 'Lasut Hanny')
    display_sentiment_analysis(df_steven, 'Steven Alfred')

# Jalankan aplikasi Streamlit
if __name__ == "__main__":
    main()
