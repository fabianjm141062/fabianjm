import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load the RUED dataset
data = pd.read_csv('datasetrued.csv')

# Select relevant columns: Population and Electricity Consumption
X = data[['Jumlah Penduduk']]
y = data['Konsumsi Listrik Per Kapita']

# Handle missing or non-numeric values
if X.isnull().values.any() or y.isnull().values.any():
    st.write("Handling missing values...")
    X = X.dropna()
    y = y.dropna()

# Train the RandomForest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Streamlit App
st.title('Electricity Consumption Prediction Based on Population by Fabian J Manoppo')

# User input for population value
population_input = st.number_input('Input Population (Jumlah Penduduk)', value=float(data['Jumlah Penduduk'].mean()))

# Make prediction
if st.button('Predict'):
    prediction = rf_model.predict([[population_input]])
    st.write(f'Predicted Electricity Consumption Per Capita: {prediction[0]} kWh')

# Feature importance (though in this case we only have one feature: Population)
importances = rf_model.feature_importances_
plt.figure(figsize=(6, 4))
plt.barh(['Jumlah Penduduk'], importances, align='center')
plt.xlabel('Feature Importance')
plt.title('Feature Importance for Electricity Consumption Prediction')
st.pyplot(plt)
