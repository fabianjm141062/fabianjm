import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np

# Load the dataset (replace 'dataset_rued_sulut.csv' with your actual dataset path)
data = pd.read_csv('dataset_rued_sulut.csv')

# Preprocessing: handle missing values, convert columns to numeric
data.fillna(0, inplace=True)  # Handle missing values
data['Jumlah Penduduk'] = pd.to_numeric(data['Jumlah Penduduk'], errors='coerce')
data['Pertumbuhan Industri'] = pd.to_numeric(data['Pertumbuhan Industri'], errors='coerce')
data['Energi Terbarukan'] = pd.to_numeric(data['Energi Terbarukan'], errors='coerce')
data['Energi Fosil'] = pd.to_numeric(data['Energi Fosil'], errors='coerce')

# Define the target variable and features
X = data[['Jumlah Penduduk', 'Pertumbuhan Industri']]
y_renewable = data['Energi Terbarukan']
y_fossil = data['Energi Fosil']

# Train a RandomForest model for both renewable and fossil energy predictions
rf_renewable = RandomForestRegressor(n_estimators=100, random_state=42)
rf_fossil = RandomForestRegressor(n_estimators=100, random_state=42)
rf_renewable.fit(X, y_renewable)
rf_fossil.fit(X, y_fossil)

# Streamlit App
st.title('Prediksi Kebutuhan Energi: Terbarukan vs Fosil by Fabian J Manoppo')

# User input for population and industry growth
population_input = st.number_input('Masukkan Jumlah Penduduk:', value=float(data['Jumlah Penduduk'].mean()))
industry_input = st.number_input('Masukkan Pertumbuhan Industri (%):', value=float(data['Pertumbuhan Industri'].mean()))

# Predict energy requirements
if st.button('Prediksi'):
    prediction_renewable = rf_renewable.predict([[population_input, industry_input]])
    prediction_fossil = rf_fossil.predict([[population_input, industry_input]])
    ratio = prediction_renewable[0] / (prediction_fossil[0] + 1e-5)  # To avoid division by zero

    st.write(f'Prediksi Energi Terbarukan: {prediction_renewable[0]:.2f} TOE')
    st.write(f'Prediksi Energi Fosil: {prediction_fossil[0]:.2f} TOE')
    st.write(f'Rasio Energi Terbarukan/Fosil: {ratio:.2f}')

    # Visualization
    labels = ['Energi Terbarukan', 'Energi Fosil']
    values = [prediction_renewable[0], prediction_fossil[0]]
    plt.figure(figsize=(8, 6))
    plt.pie(values, labels=labels, autopct='%1.1f%%', colors=['green', 'gray'])
    plt.title('Rasio Kebutuhan Energi Terbarukan vs Fosil')
    st.pyplot(plt)

# Create a scatter plot showing energy needs based on population and industry growth
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(data['Jumlah Penduduk'], data['Energi Terbarukan'], color='green', label='Energi Terbarukan')
ax.scatter(data['Jumlah Penduduk'], data['Energi Fosil'], color='gray', label='Energi Fosil')
ax.set_xlabel('Jumlah Penduduk')
ax.set_ylabel('Kebutuhan Energi (TOE)')
ax.set_title('Hubungan Kebutuhan Energi dengan Jumlah Penduduk dan Industri')
ax.legend()
st.pyplot(fig)
