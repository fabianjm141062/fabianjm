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

# Define possible values for the categorical variables
gender_options = ['Male', 'Female']
diabetes_options = ['Yes', 'No']
smoking_status_options = ['Never', 'Current']
chest_pain_options = ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain']

# Encode categorical variables for the model using the dataset
encoder_gender = LabelEncoder().fit(gender_options)
encoder_diabetes = LabelEncoder().fit(diabetes_options)
encoder_smoking_status = LabelEncoder().fit(smoking_status_options)
encoder_chest_pain = LabelEncoder().fit(chest_pain_options)

heart_data['Gender'] = encoder_gender.transform(heart_data['Gender'])
heart_data['Has Diabetes'] = encoder_diabetes.transform(heart_data['Has Diabetes'])
heart_data['Smoking Status'] = encoder_smoking_status.transform(heart_data['Smoking Status'])
heart_data['Chest Pain Type'] = encoder_chest_pain.transform(heart_data['Chest Pain Type'])
heart_data['Treatment'] = LabelEncoder().fit_transform(heart_data['Treatment'])

# Define features and target
X = heart_data.drop('Treatment', axis=1)  # Features
y = heart_data['Treatment']  # Target variable

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Streamlit app
st.title("Heart Disease Treatment Prediction")

st.write("This app predicts the type of treatment required for heart disease based on input features.")

# User input
gender = st.selectbox('Gender', gender_options)
age = st.number_input('Age', min_value=1, max_value=120, value=30)
blood_pressure = st.number_input('Blood Pressure (mmHg)', min_value=50, max_value=200, value=120)
cholesterol = st.number_input('Cholesterol (mg/dL)', min_value=100, max_value=400, value=200)
diabetes = st.selectbox('Has Diabetes', diabetes_options)
smoking_status = st.selectbox('Smoking Status', smoking_status_options)
chest_pain = st.selectbox('Chest Pain Type', chest_pain_options)

# Encode the user input using the predefined encoders
gender_encoded = encoder_gender.transform([gender])[0]
diabetes_encoded = encoder_diabetes.transform([diabetes])[0]
smoking_status_encoded = encoder_smoking_status.transform([smoking_status])[0]
chest_pain_encoded = encoder_chest_pain.transform([chest_pain])[0]

# Create input array for prediction
input_data = [[gender_encoded, age, blood_pressure, cholesterol, diabetes_encoded, smoking_status_encoded, chest_pain_encoded]]

# Make prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    treatment = LabelEncoder().fit_transform(heart_data['Treatment'])
    st.write(f"Predicted Treatment: {prediction[0]}")
