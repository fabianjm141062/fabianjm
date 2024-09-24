import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('dataset_rued_sulut _new.csv')

# Display the available columns in the dataset
st.write("Available Columns in Dataset:")
st.write(data.columns)

# Ask the user to select the column representing the year
year_column = st.selectbox("Please select the Year column:", data.columns)

# Select features for prediction (Pendapatan Per Kapita, Jumlah Penduduk, etc.)
features = st.multiselect('Select Features for Prediction (e.g., Per Capita Income, Population):', list(data.columns))

# Select the column for electricity demand as the target
electricity_demand_column = st.selectbox("Please select the Electricity Demand column:", data.columns)

# Select year for prediction
years = data[year_column].unique()
selected_year = st.selectbox('Select Year:', years)

# Filter dataset by the selected year
filtered_data = data[data[year_column] == selected_year]

# Check if the user has selected enough features for prediction
if len(features) >= 1:
    X = filtered_data[features]
    y_electricity = filtered_data[electricity_demand_column]

    # Train the RandomForest model
    rf_electricity = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_electricity.fit(X, y_electricity)

    # Input for user to predict electricity demand
    st.write(f"Enter values for the selected features for the year {selected_year}:")
    input_values = [st.number_input(f'Input {feature}', value=float(filtered_data[feature].mean())) for feature in features]

    if st.button('Predict'):
        electricity_pred = rf_electricity.predict([input_values])[0]

        # Show the prediction
        st.write(f"Predicted Electricity Demand: {electricity_pred:.2f} TOE")

        # Create a bar chart for electricity demand
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(['Predicted Electricity Demand'], [electricity_pred], color='blue', label='Electricity Demand')
        ax.set_ylabel('Energy (TOE)')
        ax.set_title(f'Electricity Demand for {selected_year}')
        ax.legend()
        st.pyplot(fig)

        # Feature importance for electricity demand
        electricity_importances = rf_electricity.feature_importances_

        # Display feature importances for electricity demand
        st.write("Feature Importance for Electricity Demand Prediction:")
        fig, ax = plt.subplots()
        ax.barh(features, electricity_importances, color='blue')
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance for Electricity Demand Prediction')
        st.pyplot(fig)
else:
    st.write("Please select at least one feature for prediction.")
