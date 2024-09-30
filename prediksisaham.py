import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
from datetime import date, timedelta

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

# Input for number of days to predict
num_days = st.sidebar.number_input("Number of future days to predict", min_value=1, max_value=365, value=60)

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

# Reshape the input to be [samples, time steps, features] for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=1, epochs=5)

# Predict on the test set
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Evaluate the model
st.subheader("Model Performance")
st.write(f"Mean Squared Error on Test Set: {np.mean((predictions - scaler.inverse_transform(y_test.reshape(-1, 1))) ** 2):.4f}")

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

# Predict future prices based on user input in 30-day intervals
st.subheader(f"Future Price Prediction for the next {num_days} days with 30-day intervals")

# Get the last `time_step` days of data for prediction
last_days = scaled_data[-time_step:]
future_predictions = []
prediction_intervals = []

# Predict in 30-day intervals
for i in range(0, num_days, 30):
    # Create temporary storage for predictions in each interval
    temp_future_predictions = []
    
    # Predict 30 days ahead in each interval
    for j in range(30):
        # Reshape the last `time_step` days into LSTM input format
        input_data = np.reshape(last_days, (1, time_step, 1))
        
        # Predict the next day
        predicted_price = model.predict(input_data)
        
        # Append the predicted price
        temp_future_predictions.append(predicted_price[0][0])
        
        # Update the input data with the predicted price for the next iteration
        last_days = np.append(last_days[1:], predicted_price)
        last_days = np.reshape(last_days, (time_step, 1))
    
    # Store predictions for each interval
    future_predictions.append(temp_future_predictions)
    prediction_intervals.append(i + 30)

# Flatten the predictions and inverse transform to get actual predicted prices
future_predictions_flat = [item for sublist in future_predictions for item in sublist]
future_predictions_flat = scaler.inverse_transform(np.array(future_predictions_flat).reshape(-1, 1))

# Create dates for the future predictions in 30-day intervals
last_date = data.index[-1]
future_dates = [last_date + timedelta(days=i) for i in range(1, len(future_predictions_flat) + 1)]

# Create a dataframe for future predictions
future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions_flat.flatten()})
st.write(future_df)

# Plot future predictions
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(future_dates, future_predictions_flat, label='Predicted Price', color='red')
ax2.set_xlabel("Date")
ax2.set_ylabel("Price in USD")
ax2.set_title(f"Predicted Stock Price of {selected_stock} for the Next {num_days} Days in 30-Day Intervals")
st.pyplot(fig2)
