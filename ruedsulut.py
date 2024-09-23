import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Load the RUED dataset
data = pd.read_csv('datasetrued.csv')  # Replace this with your actual dataset file path

# Ensure the relevant columns are numeric
data['Jumlah Penduduk'] = pd.to_numeric(data['Jumlah Penduduk'], errors='coerce')
data['Konsumsi Listrik Per Kapita'] = pd.to_numeric(data['Konsumsi Listrik Per Kapita'], errors='coerce')

# Drop rows where either column has missing or invalid data
data_clean = data[['Jumlah Penduduk', 'Konsumsi Listrik Per Kapita']].dropna()

# Separate features (X) and target (y)
X = data_clean[['Jumlah Penduduk']]
y = data_clean['Konsumsi Listrik Per Kapita']

# Train the Decision Tree model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X, y)

# Streamlit App
st.title('Electricity Consumption Prediction (Actual vs Predicted) - Decision Tree')

# User input for population value
population_input = st.number_input('Input Population (Jumlah Penduduk)', value=float(data['Jumlah Penduduk'].mean()))

# Make prediction for user input
if st.button('Predict'):
    prediction = dt_model.predict([[population_input]])
    st.write(f'Predicted Electricity Consumption Per Capita: {prediction[0]} kWh')

# Predict for the entire dataset (for comparison with actual values)
predictions = dt_model.predict(X)

# Plot Actual vs Predicted Values
plt.figure(figsize=(10, 6))
plt.plot(X, y, 'bo-', label='Actual Electricity Consumption')
plt.plot(X, predictions, 'r*-', label='Predicted Electricity Consumption')
plt.xlabel('Population (Jumlah Penduduk)')
plt.ylabel('Electricity Consumption Per Capita (kWh)')
plt.title('Actual vs Predicted Electricity Consumption')
plt.legend()
st.pyplot(plt)
