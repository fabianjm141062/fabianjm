import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Load the dataset and train the model
@st.cache_data
def load_and_train_model():
    # Load data (assuming it's already cleaned and in proper format)
    data = pd.read_csv("kerala.csv")
    
    # Extract features (monthly rainfall data) and target (flood occurrence)
    X = data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']]
    y = data['FLOODS'].apply(lambda x: 1 if x == 'YES' else 0)
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    
    return rf_model

# Load model
model = load_and_train_model()

# Streamlit App UI
st.title("Flood Prediction App by Fabian J Manoppo")
st.write("Enter the monthly rainfall data to predict whether a flood will occur.")

# Create input fields for rainfall data
rainfall_data = []
months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
for month in months:
    value = st.number_input(f'Rainfall in {month} (mm)', min_value=0.0, value=0.0)
    rainfall_data.append(value)

# Button for prediction
if st.button('Predict Flood'):
    if len(rainfall_data) == 12:
        # Make prediction
        prediction = model.predict([rainfall_data])
        result = 'YES' if prediction[0] == 1 else 'NO'
        st.write(f"Flood Prediction: **{result}**")
    else:
        st.write("Please enter rainfall data for all 12 months.")

