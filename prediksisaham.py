import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from datetime import timedelta
import os

# Optional: Force TensorFlow to use CPU (if there are GPU-related issues)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Function to fetch stock data from Yahoo Finance
def load_stock_data(ticker):
    data = yf.download(ticker, start="2010-01-01", end="2024-12-31")
    return data

# Title of the app
st.title('Stock Price Prediction using Deep Learning oleh Fabian J Manoppo')

# Sidebar stock selection
st.sidebar.subheader("Stock Selection")

# Predefined Indonesian stock options for easy selection
stock_options = {
    'Apple (AAPL)': 'AAPL',
    'Telkom Indonesia (TLKM.JK)': 'TLKM.JK',
    'Bank Rakyat Indonesia (BBRI.JK)': 'BBRI.JK',
    'Bank Central Asia (BBCA.JK)': 'BBCA.JK'
}
selected_stock = st.sidebar.selectbox("Choose a stock", list(stock_options.keys()))
stock_ticker = stock_options[selected_stock]

# Number of future days is fixed at 30 days
num_days = 30

# Load stock data
st.subheader(f"Stock data for {selected_stock}")
data = load_stock_data(stock_ticker)

# Display the stock data
st.write(data.tail())

# Preprocessing the data
data_close = data[['Close']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_close)

# Split the data into training and test sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Function to create dataset for LSTM
def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

# Create the LSTM training dataset
time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Check the shapes of X_train and y_train
st.write(f"X_train shape: {X_train.shape}")
st.write(f"y_train shape: {y_train.shape}")

# Reshape the input to be [samples, time steps, features] for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build a simpler LSTM model to avoid resource issues
model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(time_step, 1)))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model with error handling
try:
    model.fit(X_train, y_train, batch_size=5, epochs=2)
    st.success("Model training completed successfully.")
except Exception as e:
    st.error(f"Error during model training: {e}")

# Predict on the test set
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Evaluate the model
st.subheader("Model Performance")
try:
    mse = np.mean((predictions - scaler.inverse_transform(y_test.reshape(-1, 1))) ** 2)
    st.write(f"Mean Squared Error on Test Set: {mse:.4f}")
except Exception as e:
    st.error(f"Error during model evaluation: {e}")

# Plot actual vs predicted stock prices
st.subheader(f"Stock closing price prediction for {selected_stock}")
fig, ax = plt.subplots(figsize=(10, 6))
train_data_plot = data['Close'][:train_size]
test_data_plot = data['Close'][train_size:]
ax.plot(data.index[:train_size], train_data_plot, label='Training Data', color='blue')
ax.plot(data.index[train_size:], test_data_plot, label='Actual Prices', color='green')
ax.plot(data.index[train_size + time_step + 1:], predictions, label='Predicted Prices', color='red')
ax.set_xlabel("Date")
ax.set_ylabel("Price in USD")
ax.set_title(f"Actual vs Predicted Prices for {selected_stock}")
ax.legend()
st.pyplot(fig)

# Predict future prices for the next 30 days
st.subheader(f"Future Price Prediction for the next {num_days} days")

# Get the last `time_step` days of data for prediction
last_days = scaled_data[-time_step:]
future_predictions = []

# Predict for 30 days
for i in range(num_days):
    # Reshape the last `time_step` days into LSTM input format
    input_data = np.reshape(last_days, (1, time_step, 1))

    # Predict the next day
    predicted_price = model.predict(input_data)

    # Append the predicted price to future_predictions
    future_predictions.append(predicted_price[0][0])

    # Update the input data with the predicted price for the next iteration
    last_days = np.append(last_days[1:], predicted_price)
    last_days = np.reshape(last_days, (time_step, 1))

# Inverse transform
