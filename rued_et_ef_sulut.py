import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('dataset_rued_sulut _new.csv')

# Display the available columns in the dataset
st.write("Available Columns in Dataset:")
st.write(data.columns)

# Ask the user to select the column representing the year if 'Year' is not found
year_column = st.selectbox("Please select the Year column:", data.columns)

# Ensure the selected column is numeric and handle missing values
data[year_column] = pd.to_numeric(data[year_column], errors='coerce')

# Select features for prediction
features = st.multiselect('Select Features for Prediction:', list(data.columns))

# Select year for prediction
years = data[year_column].unique()
selected_year = st.selectbox('Select Year:', years)

# Filter dataset by the selected year
filtered_data = data[data[year_column] == selected_year]

# Check if the user has selected enough features for prediction
if len(features) >= 1:
    X = filtered_data[features]
    y_renewable = filtered_data['Energi Terbarukan']
    y_fossil = filtered_data['Energi Fosil']

    # Train the RandomForest models
    rf_renewable = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_fossil = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_renewable.fit(X, y_renewable)
    rf_fossil.fit(X, y_fossil)

    # Input for user to predict energy needs
    st.write(f"Enter values for the selected features for the year {selected_year}:")
    input_values = [st.number_input(f'Input {feature}', value=float(filtered_data[feature].mean())) for feature in features]

    if st.button('Predict'):
        renewable_pred = rf_renewable.predict([input_values])[0]
        fossil_pred = rf_fossil.predict([input_values])[0]

        # Show the predictions
        st.write(f"Predicted Renewable Energy Need: {renewable_pred:.2f} TOE")
        st.write(f"Predicted Fossil Energy Need: {fossil_pred:.2f} TOE")

        # Create a pie chart for energy sources
        labels = ['Energi Terbarukan', 'Energi Fosil']
        values = [renewable_pred, fossil_pred]
        plt.figure(figsize=(6, 6))
        plt.pie(values, labels=labels, autopct='%1.1f%%', colors=['green', 'gray'])
        plt.title(f'Energy Source Distribution for {selected_year}')
        st.pyplot(plt)

        # Create a bar chart for electricity demand and energy sources
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(['Electricity Demand'], [filtered_data['Electricity Demand'].mean()], color='blue', label='Electricity Demand')
        ax.bar(['Energi Terbarukan'], [renewable_pred], color='green', label='Energi Terbarukan')
        ax.bar(['Energi Fosil'], [fossil_pred], color='gray', label='Energi Fosil')
        ax.set_ylabel('Energy (TOE)')
        ax.set_title(f'Electricity Demand and Energy Sources for {selected_year}')
        ax.legend()
        st.pyplot(fig)
else:
    st.write("Please select at least one feature for prediction.")
