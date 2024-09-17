import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load the dataset
@st.cache
def load_data():
    return pd.read_csv('heart_attack_dataset.csv')

# Load and preprocess data
heart_data = load_data()

# Encode categorical variables
encoder = LabelEncoder()
heart_data['Gender'] = encoder.fit_transform(heart_data['Gender'])
heart_data['Has Diabetes'] = encoder.fit_transform(heart_data['Has Diabetes'])
heart_data['Smoking Status'] = encoder.fit_transform(heart_data['Smoking Status'])
heart_data['Chest Pain Type'] = encoder.fit_transform(heart_data['Chest Pain Type'])
heart_data['Treatment'] = encoder.fit_transform(heart_data['Treatment'])

# Define features and target
X = heart_data.drop('Treatment', axis=1)  # Features
y = heart_data['Treatment']  # Target variable

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Streamlit app
st.title("Heart Disease Treatment Prediction by Fabian J Manoppo")

st.write("This app predicts the type of treatment required for heart disease based on input features.")

# User input
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age', min_value=1, max_value=120, value=30)
blood_pressure = st.number_input('Blood Pressure (mmHg)', min_value=50, max_value=200, value=120)
cholesterol = st.number_input('Cholesterol (mg/dL)', min_value=100, max_value=400, value=200)
diabetes = st.selectbox('Has Diabetes', ['Yes', 'No'])
smoking_status = st.selectbox('Smoking Status', ['Never', 'Current'])
chest_pain = st.selectbox('Chest Pain Type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain'])

# Encoding user input for prediction
gender = encoder.transform([gender])[0]
diabetes = encoder.transform([diabetes])[0]
smoking_status = encoder.transform([smoking_status])[0]
chest_pain = encoder.transform([chest_pain])[0]

# Create input array for prediction
input_data = [[gender, age, blood_pressure, cholesterol, diabetes, smoking_status, chest_pain]]

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    treatment = encoder.inverse_transform(prediction)
    st.write(f"Predicted Treatment: {treatment[0]}")

# Run Streamlit app by typing the following in your terminal:
# streamlit run your_script_name.py
