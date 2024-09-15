import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Function to fetch stock data from Yahoo Finance
def load_stock_data(ticker):
    data = yf.download(ticker, start="2010-01-01", end="2024-09-15")
    return data

# Title of the app
st.title('Stock Price Prediction App oleh Fabian J Manoppo')

# Input for stock selection (both international and Indonesian markets)
st.sidebar.subheader("Stock Selection")
stock_ticker = st.sidebar.text_input("Enter the stock ticker (e.g., AAPL for Apple, TLKM.JK for Telkom Indonesia)", value="AAPL")

# Load stock data
st.subheader(f"Stock data for {stock_ticker}")
data = load_stock_data(stock_ticker)

# Display the stock data
st.write(data.tail())

# Plot stock closing price history
st.subheader(f"Stock closing price history for {stock_ticker}")
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(data['Close'], label='Closing Price')
ax.set_xlabel("Date")
ax.set_ylabel("Price in USD")
ax.set_title(f"Closing Price of {stock_ticker} Over Time")
st.pyplot(fig)

# Feature Engineering - Creating new features (e.g., moving averages)
data['MA50'] = data['Close'].rolling(50).mean()  # 50-day moving average
data['MA200'] = data['Close'].rolling(200).mean()  # 200-day moving average
data = data.dropna()  # Remove any rows with missing values due to moving averages

# Preparing the dataset for machine learning
X = data[['MA50', 'MA200']]
y = data['Close']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training - Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predicting on the test set
y_pred = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
st.subheader("Model Performance")
st.write(f"Mean Absolute Error: {mae:.2f}")
st.write(f"Mean Squared Error: {mse:.2f}")

# Predict future price (for example, predicting the next 30 days)
st.subheader("Future Price Prediction (next 30 days)")
last_ma50 = data['MA50'].iloc[-1]
last_ma200 = data['MA200'].iloc[-1]
future_ma50 = np.tile(last_ma50, 30)
future_ma200 = np.tile(last_ma200, 30)
future_X = np.column_stack((future_ma50, future_ma200))

future_predictions = model.predict(future_X)
future_dates = pd.date_range(start=data.index[-1], periods=30, freq='D')
future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions})
st.write(future_df)

# Plot future predictions
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(future_dates, future_predictions, label='Predicted Price')
ax2.set_xlabel("Date")
ax2.set_ylabel("Price in USD")
ax2.set_title(f"Predicted Stock Price of {stock_ticker} for the Next 30 Days")
st.pyplot(fig2)
