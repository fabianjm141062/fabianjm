import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler (once saved)
# For deployment, you will need to save your model and scaler to disk and load them in the app
# model = pickle.load(open('logistic_model.pkl', 'rb'))
# scaler = pickle.load(open('scaler.pkl', 'rb'))

# For demo, we'll use the trained model and scaler from above
# In actual use case, you'd load them from the pickle files

def predict_kidney_stone(gravity, ph, osmo, cond, urea, calc):
    # Scale the input features
    scaled_data = scaler.transform([[gravity, ph, osmo, cond, urea, calc]])
    
    # Predict the result using the loaded model
    prediction = model.predict(scaled_data)
    
    # Return the result
    return 'Kidney Stone Present' if prediction[0] == 1 else 'No Kidney Stone'

# Streamlit App
st.title("Kidney Stone Prediction with AI by Fabian J Manoppo")

# Collect user input features for prediction
gravity = st.number_input('Urine Specific Gravity', min_value=1.000, max_value=1.030, value=1.010)
ph = st.number_input('Urine pH', min_value=4.0, max_value=8.0, value=5.5)
osmo = st.number_input('Urine Osmolality', min_value=0, max_value=1000, value=400)
cond = st.number_input('Urine Conductivity', min_value=0.0, max_value=100.0, value=20.0)
urea = st.number_input('Urea Concentration', min_value=0, max_value=1000, value=300)
calc = st.number_input('Calcium Concentration', min_value=0.0, max_value=10.0, value=2.5)

# Prediction button
if st.button('Predict'):
    result = predict_kidney_stone(gravity, ph, osmo, cond, urea, calc)
    st.success(f'Result: {result}')
