import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the RUED dataset
data = pd.read_csv('datasetrued.csv')

# Ensure the relevant columns are numeric
data['Jumlah Penduduk'] = pd.to_numeric(data['Jumlah Penduduk'], errors='coerce')
data['Konsumsi Listrik Per Kapita'] = pd.to_numeric(data['Konsumsi Listrik Per Kapita'], errors='coerce')

# Drop rows where either column has missing or invalid data
data_clean = data[['Jumlah Penduduk', 'Konsumsi Listrik Per Kapita']].dropna()

# Separate features (X) and target (y)
X = data_clean[['Jumlah Penduduk']]
y = data_clean['Konsumsi Listrik Per Kapita']

# Train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X, y)

# Streamlit App
st.title('Electricity Consumption Prediction Based on Population in North Sulawesi by Fabian J Manoppo')

# User input for population value
population_input = st.number_input('Input Population (Jumlah Penduduk)', value=float(data['Jumlah Penduduk'].mean()))

# Make prediction
if st.button('Predict'):
    prediction = lr_model.predict([[population_input]])
    st.write(f'Predicted Electricity Consumption Per Capita: {prediction[0]} kWh')

# Plot the regression line
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, lr_model.predict(X), color='red', label='Regression Line')
plt.xlabel('Population (Jumlah Penduduk)')
plt.ylabel('Electricity Consumption Per Capita (kWh)')
plt.title('Linear Regression: Population vs Electricity Consumption')
plt.legend()
st.pyplot(plt)
