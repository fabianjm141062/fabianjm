import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the dataset (Assuming the file is stored locally or from an online repo)
@st.cache
def load_data():
    return pd.read_csv('hypertension_data.csv')

# Load and preprocess data
data = load_data()

# Display dataset overview
st.title("Hypertension Prediction App")
st.write("This app predicts whether an individual has hypertension based on input features.")

st.write("### Dataset Overview")
st.write(data.head())

# Preprocessing: assuming the dataset has categorical and numerical features
# Replace 'Hypertension' with the actual target column in your dataset
target = 'Hypertension'  
X = data.drop(target, axis=1)  # Features
y = data[target]  # Target

# Encoding categorical variables if present
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = LabelEncoder().fit_transform(X[col])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Accuracy of the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# User input form for prediction
st.write("### Enter Patient Data for Hypertension Prediction oleh Fabian J Manoppo")
age = st.number_input('Age', min_value=0, max_value=120, value=30)
blood_pressure = st.number_input('Blood Pressure (mmHg)', min_value=80, max_value=200, value=120)
cholesterol = st.number_input('Cholesterol (mg/dL)', min_value=100, max_value=400, value=200)
# Add more features depending on your dataset

# Collect input data for prediction
input_data = pd.DataFrame([[age, blood_pressure, cholesterol]], columns=['Age', 'Blood Pressure', 'Cholesterol'])

# Make prediction based on input
if st.button('Predict'):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.write("Prediction: The patient is likely to have hypertension.")
    else:
        st.write("Prediction: The patient is unlikely to have hypertension.")
