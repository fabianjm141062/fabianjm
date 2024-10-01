import tweepy
import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# ---------------------------------
# Twitter API Setup
# ---------------------------------
consumer_key = 'YOUR_TWITTER_CONSUMER_KEY'
consumer_secret = 'YOUR_TWITTER_CONSUMER_SECRET'
access_token = 'YOUR_TWITTER_ACCESS_TOKEN'
access_token_secret = 'YOUR_TWITTER_ACCESS_TOKEN_SECRET'

# Authentikasi ke Twitter API
auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
twitter_api = tweepy.API(auth)

# ---------------------------------
# Facebook & Instagram API Setup
# ---------------------------------
fb_access_token = 'YOUR_FACEBOOK_ACCESS_TOKEN'
ig_user_id = 'YOUR_INSTAGRAM_USER_ID'  # Instagram Business Account ID

# Inisialisasi VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# ---------------------------------
# Twitter Function
# ---------------------------------
def get_twitter_data(keyword, count=100):
    tweets = twitter_api.search(q=keyword, count=count, lang="id", tweet_mode="extended")
    tweet_list = []
    for tweet in tweets:
        sentiment = analyzer.polarity_scores(tweet.full_text)['compound']
        if sentiment >= 0.05:
            sentiment_label = "Positif"
        elif sentiment <= -0.05:
            sentiment_label = "Negatif"
        else:
            sentiment_label = "Netral"
        tweet_list.append([tweet.full_text, tweet.favorite_count, tweet.retweet_count, sentiment_label])
    
    return pd.DataFrame(tweet_list, columns=['Post', 'Likes', 'Shares', 'Sentiment'])

# ---------------------------------
# Facebook Function
# ---------------------------------
def get_facebook_data(page_id):
    url = f"https://graph.facebook.com/{page_id}/posts?access_token={fb_access_token}"
    response = requests.get(url)
    posts = response.json()['data']
    
    post_list = []
    for post in posts:
        post_id = post['id']
        post_url = f"https://graph.facebook.com/{post_id}?fields=message,likes.summary(true),shares&access_token={fb_access_token}"
        post_data = requests.get(post_url).json()
        message = post_data.get('message', 'No text')
        likes = post_data['likes']['summary']['total_count']
        shares = post_data.get('shares', {}).get('count', 0)
        sentiment = analyzer.polarity_scores(message)['compound']
        if sentiment >= 0.05:
            sentiment_label = "Positif"
        elif sentiment <= -0.05:
            sentiment_label = "Negatif"
        else:
            sentiment_label = "Netral"
        post_list.append([message, likes, shares, sentiment_label])
    
    return pd.DataFrame(post_list, columns=['Post', 'Likes', 'Shares', 'Sentiment'])

# ---------------------------------
# Instagram Function
# ---------------------------------
def get_instagram_data(user_id):
    url = f"https://graph.instagram.com/{user_id}/media?fields=id,caption,like_count&access_token={fb_access_token}"
    response = requests.get(url)
    posts = response.json()['data']
    
    post_list = []
    for post in posts:
        post_url = f"https://graph.instagram.com/{post['id']}?fields=caption,like_count,media_url&access_token={fb_access_token}"
        post_data = requests.get(post_url).json()
        caption = post_data.get('caption', 'No caption')
        likes = post_data['like_count']
        sentiment = analyzer.polarity_scores(caption)['compound']
        if sentiment >= 0.05:
            sentiment_label = "Positif"
        elif sentiment <= -0.05:
            sentiment_label = "Negatif"
        else:
            sentiment_label = "Netral"
        post_list.append([caption, likes, sentiment_label])
    
    return pd.DataFrame(post_list, columns=['Post', 'Likes', 'Sentiment'])

# ---------------------------------
# Main Program - Fetch Data
# ---------------------------------

# Twitter Data
df_yulius_twitter = get_twitter_data('Yulius Victor', 100)
df_lasut_twitter = get_twitter_data('Lasut Hanny', 100)
df_steven_twitter = get_twitter_data('Steven Alfred', 100)

# Facebook Data
df_yulius_fb = get_facebook_data('YuliusVictorPageID')
df_lasut_fb = get_facebook_data('LasutHannyPageID')
df_steven_fb = get_facebook_data('StevenAlfredPageID')

# Instagram Data
df_steven_ig = get_instagram_data(ig_user_id)

# ---------------------------------
# Menggabungkan Data dan Visualisasi
# ---------------------------------

# Gabungkan data dari platform berbeda
df_combined = pd.concat([df_yulius_twitter, df_lasut_twitter, df_steven_twitter, df_yulius_fb, df_lasut_fb, df_steven_fb, df_steven_ig])

# Visualisasi Sentimen
sentiment_counts = df_combined['Sentiment'].value_counts()
sentiment_counts.plot(kind='bar')
plt.title('Analisis Sentimen Kandidat di Media Sosial')
plt.show()

# Visualisasi Popularitas
df_combined.groupby('Sentiment').size().plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Distribusi Sentimen Media Sosial')
plt.show()
