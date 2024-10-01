import tweepy
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import streamlit as st

# Twitter API v2 credentials
bearer_token = 'YOUR_TWITTER_BEARER_TOKEN'

# Inisialisasi Tweepy Client untuk Twitter API v2
client = tweepy.Client(bearer_token=bearer_token)

# Inisialisasi VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Fungsi untuk mendapatkan tweet dan melakukan analisis sentimen secara real-time
def get_twitter_data_real_time(keyword, count=100):
    query = f"{keyword} -is:retweet lang:id"
    tweets = client.search_recent_tweets(query=query, max_results=count, tweet_fields=['public_metrics', 'created_at'])
    
    tweet_list = []
    for tweet in tweets.data:
        sentiment = analyzer.polarity_scores(tweet.text)['compound']
        if sentiment >= 0.05:
            sentiment_label = "Positif"
        elif sentiment <= -0.05:
            sentiment_label = "Negatif"
        else:
            sentiment_label = "Netral"
        tweet_list.append([tweet.text, tweet.public_metrics['like_count'], tweet.public_metrics['retweet_count'], sentiment_label, tweet.created_at])
    
    return pd.DataFrame(tweet_list, columns=['Post', 'Likes', 'Retweets', 'Sentiment', 'Timestamp'])

# Fungsi untuk menampilkan hasil analisis dalam Streamlit
def display_sentiment_analysis(df, candidate_name):
    st.subheader(f"Analisis Sentimen untuk {candidate_name}")
    
    # Hitung persentase sentimen
    sentiment_counts = df['Sentiment'].value_counts(normalize=True) * 100
    st.bar_chart(sentiment_counts)
    
    # Tampilkan post dengan jumlah like tertinggi
    st.subheader(f"Post dengan Like Tertinggi untuk {candidate_name}")
    most_liked = df[df['Likes'] == df['Likes'].max()]
    st.write(most_liked[['Post', 'Likes', 'Retweets', 'Sentiment', 'Timestamp']])

# Fungsi utama untuk Streamlit
def main():
    st.title("Real-time Analisis Sentimen & Popularitas Kandidat Gubernur Sulawesi Utara")
    
    st.write("Data ini diambil secara real-time dari Twitter menggunakan API Twitter v2.")
    
    # Input pengguna untuk jumlah tweet dan kata kunci kandidat
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
