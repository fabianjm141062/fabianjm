import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Step 1: Generate synthetic dataset (or replace with real data)
def generate_data():
    np.random.seed(42)
    wind_speed = np.random.uniform(5, 25, 500)  # Wind speed (m/s)
    fetch_distance = np.random.uniform(50, 500, 500)  # Fetch distance (km)
    duration = np.random.uniform(1, 12, 500)  # Duration (hours)
    wave_height = 0.005 * wind_speed**2 * fetch_distance  # Simplified wave height calculation
    return pd.DataFrame({
        'Wind_Speed': wind_speed,
        'Fetch_Distance': fetch_distance,
        'Duration': duration,
        'Wave_Height': wave_height
    })

# Step 2: Load and train the model
@st.cache_data
def load_model():
    data = generate_data()
    X = data[['Wind_Speed', 'Fetch_Distance', 'Duration']]
    y = data['Wave_Height']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

# Load the model and data
model, X_test, y_test = load_model()

# Streamlit App Layout
st.title("Wave Height Prediction Based on Wind Data")

# Input from the user
st.subheader("Enter Wind Data")
wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, value=10.0)
fetch_distance = st.number_input("Fetch Distance (km)", min_value=0.0, value=100.0)
duration = st.number_input("Wind Duration (hours)", min_value=0.0, value=5.0)

# Predict button
if st.button('Predict Wave Height'):
    # Prepare input data for prediction
    input_data = np.array([[wind_speed, fetch_distance, duration]])
    
    # Make the prediction
    predicted_wave_height = model.predict(input_data)[0]
    
    # Show the result
    st.write(f"Predicted Wave Height: {predicted_wave_height:.2f} meters")

    # Plot actual vs predicted wave height
    st.subheader("Comparison of Actual and Predicted Wave Heights")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X_test['Wind_Speed'], y_test, color='blue', label='Actual Wave Height')
    ax.scatter(X_test['Wind_Speed'], model.predict(X_test), color='red', label='Predicted Wave Height')
    ax.set_xlabel('Wind Speed (m/s)')
    ax.set_ylabel('Wave Height (meters)')
    ax.legend()
    st.pyplot(fig)
