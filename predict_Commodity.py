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

# Function to fetch commodity data from Yahoo Finance (with reduced data range)
def load_commodity_data(ticker):
    data = yf.download(ticker, start="2019-01-01", end="2024-12-31")  # Reduced range for commodity
    return data

# Title of the app
st.title('Commodity Price Prediction using Deep Learning by Fabian J Manoppo')

# Sidebar commodity selection
st.sidebar.subheader("Commodity Selection")

# Allow the user to input any commodity ticker (e.g., GC=F for Gold, CL=F for Crude Oil)
commodity_ticker = st.sidebar.text_input("Enter the commodity ticker (e.g., GC=F for Gold, CL=F for Crude Oil)", value="GC=F")

# Define the number of future days to predict
num_days = st.sidebar.number_input("Number of future days to predict", min_value=1, max_value=365, value=60)

# Automatically load commodity data and preprocess it
st.subheader(f"Commodity data for {commodity_ticker}")
try:
    # Load commodity data
    data = load_commodity_data(commodity_ticker)
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

    # Reshape the input to be [samples, time steps, features] for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Build a simpler LSTM model to avoid resource issues
    model = Sequential()
    model.add(LSTM(32, return_sequences=False, input_shape=(time_step, 1)))  # Correct input shape (time_steps, features)
    model.add(Dense(16))  # Simpler model with fewer neurons
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    with st.spinner('Training the model...'):
        model.fit(X_train, y_train, batch_size=10, epochs=2)  # Reduced batch size and epochs
        st.success("Model training completed successfully.")

    # Prediction button to trigger predictions
    if st.button('Run Predictions'):
        # Predict on the test set
        predictions = model.predict(X_test)
        predictions = scaler.inverse_transform(predictions)

        # Adjust the length of the actual test data index to match the predicted data
        test_data_index = data.index[train_size + time_step + 1: train_size + time_step + 1 + len(predictions)]

        # Plot the predicted prices
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(data.index[train_size:], scaler.inverse_transform(test_data), label='Actual Prices', color='green')
        ax.plot(test_data_index, predictions, label='Predicted Prices', color='red')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price in USD")
        ax.legend()
        st.pyplot(fig)

        # Future Price Prediction for the next `num_days`
        st.subheader(f"Future Price Prediction for the next {num_days} days")

        # Get the last `time_step` days of data for prediction
        last_days = scaled_data[-time_step:]
        future_predictions = []

        # Predict for `num_days`
        for i in range(num_days):
            input_data = np.reshape(last_days, (1, time_step, 1))
            predicted_price = model.predict(input_data)
            future_predictions.append(predicted_price[0][0])
            last_days = np.append(last_days[1:], predicted_price)
            last_days = np.reshape(last_days, (time_step, 1))

        # Inverse transform to get actual predicted prices
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        # Create dates for the future predictions
        last_date = data.index[-1]
        f
