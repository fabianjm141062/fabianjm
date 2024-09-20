import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load the RUED dataset
data = pd.read_csv('datasetrued.csv')

# Handle missing values if any
data = data.dropna()

# Separate features and target
X = data.drop(columns=['Year'])
y = data['Year']

# Train the RandomForest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Streamlit App
st.title('RUED Prediction Using RandomForest')

# User inputs for all features
input_data = []
for feature in X.columns:
    value = st.number_input(f'Input {feature}', value=float(data[feature].mean()))
    input_data.append(value)

# Make prediction
if st.button('Predict'):
    prediction = rf_model.predict([input_data])
    st.write(f'Predicted RUED Year: {prediction[0]}')

# Feature importance
importances = rf_model.feature_importances_
indices = range(len(importances))

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(X.columns, importances, align='center')
plt.xlabel('Feature Importance')
plt.title('Feature Importances in RUED Prediction')
st.pyplot(plt)