import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Load the dataset
def load_data():
    data = pd.read_csv('kidney_disease.csv')  # Make sure to adjust the path
    num_cols = data.select_dtypes(include=['float64', 'int64']).columns
    cat_cols = data.select_dtypes(include=['object']).columns
    
    # Handling missing values
    imputer_num = SimpleImputer(strategy='mean')
    imputer_cat = SimpleImputer(strategy='most_frequent')
    
    data[num_cols] = imputer_num.fit_transform(data[num_cols])
    data[cat_cols] = imputer_cat.fit_transform(data[cat_cols])
    
    # Encoding categorical variables
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        label_encoders[col] = le
    
    return data, label_encoders

data, label_encoders = load_data()

# Define input features
features = data.drop(columns=['id', 'classification'])
target = data['classification']

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(features, target)

# Streamlit interface
st.title("Kidney Disease Prediction by Fabian J Manoppo")

def user_input():
    age = st.number_input("Age", min_value=0, max_value=100, value=50)
    bp = st.number_input("Blood Pressure", min_value=50, max_value=180, value=80)
    sg = st.number_input("Specific Gravity", min_value=1.005, max_value=1.025, value=1.020, step=0.001)
    al = st.number_input("Albumin", min_value=0, max_value=5, value=1)
    su = st.number_input("Sugar", min_value=0, max_value=5, value=0)
    rbc = st.selectbox("Red Blood Cells", ['normal', 'abnormal'])
    pc = st.selectbox("Pus Cell", ['normal', 'abnormal'])
    pcc = st.selectbox("Pus Cell Clumps", ['present', 'notpresent'])
    ba = st.selectbox("Bacteria", ['present', 'notpresent'])
    bgr = st.number_input("Blood Glucose Random", min_value=70, max_value=500, value=120)
    bu = st.number_input("Blood Urea", min_value=10, max_value=150, value=50)
    sc = st.number_input("Serum Creatinine", min_value=0.4, max_value=15.0, value=1.2)
    
    # Convert input into a DataFrame
    data_input = pd.DataFrame({
        'age': [age],
        'bp': [bp],
        'sg': [sg],
        'al': [al],
        'su': [su],
        'rbc': [label_encoders['rbc'].transform([rbc])[0]],
        'pc': [label_encoders['pc'].transform([pc])[0]],
        'pcc': [label_encoders['pcc'].transform([pcc])[0]],
        'ba': [label_encoders['ba'].transform([ba])[0]],
        'bgr': [bgr],
        'bu': [bu],
        'sc': [sc],
    })
    
    # Ensure the input matches the model's feature names
    data_input = data_input.reindex(columns=features.columns, fill_value=0)
    
    return data_input

# Predict based on user input
input_data = user_input()

if st.button("Predict"):
    prediction = model.predict(input_data)
    prediction_label = 'ckd' if prediction[0] == 1 else 'notckd'
    st.write(f"Prediction: {prediction_label}")
