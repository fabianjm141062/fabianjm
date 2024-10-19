import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
from datetime import timedelta
import os

# Optional: Force TensorFlow to use CPU (if there are GPU-related issues)
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Function to fetch commodity/currency data from Yahoo Finance with fallback
def load_commodity_data(ticker, fallback_ticker=None):
    try:
        data = yf.download(ticker, start="2020-01-01", end="2024-12-31")  # Shortened range to ensure availability
        if data.empty:
            raise ValueError(f"No data found for {ticker}.")
        return data
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {e}")
        if fallback_ticker:
            st.warning(f"Trying fallback ticker: {fallback_ticker}")
            return load_commodity_data(fallback_ticker)  # Retry with fallback ticker
        else:
            return None

# Title of the app
st.title('Commodity & Currency Price Prediction with Economic Indicators by Fabian J Manoppo')

# Sidebar commodity selection
st.sidebar.subheader("Commodity/Currency Selection")

# Allow the user to select a commodity/currency ticker from the predefined list
commodity_options = {
    "Gold": "GC=F",
    "Crude Oil": "CL=F",
    "Palm Oil": "POF=F",
    "USD/IDR": "USDIDR=X",
    "USD/EUR": "EURUSD=X"
}
commodity_name = st.sidebar.selectbox("Select a commodity/currency", list(commodity_options.keys()))
commodity_ticker = commodity_options[commodity_name]

# Define a fallback ticker if needed (e.g., USD/IDR may fall back to USD/EUR)
fallback_tickers = {
    "USD/IDR": "EURUSD=X"  # Fallback for USD/IDR
}

# Define the number of future days to predict
num_days = st.sidebar.number_input("Number of future days to predict", min_value=1, max_value=365, value=60)

# Load commodity/currency data with a fallback mechanism
st.subheader(f"Data for {commodity_name}")
data = load_commodity_data(commodity_ticker, fallback_ticker=fallback_tickers.get(commodity_name))
if data is not None:
    try:
        st.write(data.tail())

        # Add volume data to the model (if available, otherwise just the 'Close' prices)
        if 'Volume' in data.columns:
            data_features = data[['Close', 'Volume']].values
        else:
            # Some currency pairs or commodities might not have volume data, use just 'Close'
            data_features = data[['Close']].values

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data_features)

        # Split the data into training and test sets
        train_size = int(len(scaled_data) * 0.8)
        train_data = scaled_data[:train_size]
        test_data = scaled_data[train_size:]

        # Function to create dataset for LSTM
        def create_dataset(dataset, time_step=60):
            X, y = [], []
            for i in range(len(dataset) - time_step - 1):
                X.append(dataset[i:(i + time_step), :])  # Use all available features (Close, Volume)
                y.append(dataset[i + time_step, 0])  # Predict only 'Close' price
            return np.array(X), np.array(y)

        # Create the LSTM training dataset
        time_step = 60
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, y_test = create_dataset(test_data, time_step)

        # Reshape the input to be [samples, time steps, features] for LSTM
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

        # Build a more robust LSTM model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(time_step, X_train.shape[2])))  # Increased LSTM layers and units
        model.add(Dropout(0.2))  # Adding dropout for regularization
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(25))
        model.add(Dense(1))

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model with more epochs for better results
        with st.spinner('Training the model...'):
            model.fit(X_train, y_train, batch_size=16, epochs=10)  # Increased epochs for better learning
            st.success("Model training completed successfully.")

        # Prediction button to trigger predictions
        if st.button('Run Predictions'):
            # Predict on the test set
            predictions = model.predict(X_test)
            # If Volume is not used, avoid scaling the second column
            if 'Volume' in data.columns:
                predictions = scaler.inverse_transform(np.concatenate((predictions, np.zeros((predictions.shape[0], 1))), axis=1))[:, 0]
            else:
                predictions = scaler.inverse_transform(predictions)

            # Adjust the length of the actual test data index to match the predicted data
            test_data_index = data.index[train_size + time_step + 1: train_size + time_step + 1 + len(predictions)]

            # Plot the predicted prices
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(data.index[train_size:], scaler.inverse_transform(test_data)[:, 0], label='Actual Prices', color='green')
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
                input_data = np.reshape(last_days, (1, time_step, X_train.shape[2]))
                predicted_price = model.predict(input_data)[0][0]

                # If using Volume, append correctly shaped array; otherwise, just the price
                if 'Volume' in data.columns:
                    # Create a 2D array with predicted price and dummy value for volume
                    last_days = np.append(last_days[1:], np.array([[predicted_price, 0]]), axis=0)
                else:
                    last_days = np.append(last_days[1:], np.array([[predicted_price]]), axis=0)

                future_predictions.append(predicted_price)

            # Inverse transform to get actual predicted prices
            future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

            # Create dates for the future predictions
            last_date = data.index[-1]
            future_dates = [last_date + timedelta(days=i) for i in range(1, num_days + 1)]

            # Plot future predictions
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.plot(future_dates, future_predictions, label='Predicted Prices', color='red')
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Price in USD")
            ax2.legend()
            st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error processing data: {e}")
