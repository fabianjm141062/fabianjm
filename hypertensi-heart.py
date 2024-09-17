import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load datasets
# Replace these with actual URLs of your datasets
hypertension_url = "URL_OF_HYPERTENSION_DATASET"  # Replace with the actual URL for hypertension dataset
heart_disease_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Load the datasets
hypertension_data = pd.read_csv(hypertension_url)
heart_disease_data = pd.read_csv(heart_disease_url, header=None)

# Preprocess the datasets
# Preprocessing Hypertension Dataset (assuming 'Target' column exists for hypertension)
hypertension_data.fillna(hypertension_data.mean(), inplace=True)
X_hyp = hypertension_data.drop('Target', axis=1)  # Replace 'Target' with the actual target column name
y_hyp = hypertension_data['Target']

# Preprocessing Heart Disease Dataset
heart_disease_data.columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
heart_disease_data.fillna(heart_disease_data.mean(), inplace=True)
X_hd = heart_disease_data.drop('target', axis=1)
y_hd = heart_disease_data['target']

# Train-test split
X_hyp_train, X_hyp_test, y_hyp_train, y_hyp_test = train_test_split(X_hyp, y_hyp, test_size=0.2, random_state=42)
X_hd_train, X_hd_test, y_hd_train, y_hd_test = train_test_split(X_hd, y_hd, test_size=0.2, random_state=42)

# Train Random Forest models
model_hyp = RandomForestClassifier()
model_hd = RandomForestClassifier()

model_hyp.fit(X_hyp_train, y_hyp_train)
model_hd.fit(X_hd_train, y_hd_train)

# Evaluate models
hyp_pred = model_hyp.predict(X_hyp_test)
hd_pred = model_hd.predict(X_hd_test)

hyp_accuracy = accuracy_score(y_hyp_test, hyp_pred)
hd_accuracy = accuracy_score(y_hd_test, hd_pred)

print(f"Hypertension Prediction Model Accuracy: {hyp_accuracy * 100:.2f}%")
print(f"Heart Disease Prediction Model Accuracy: {hd_accuracy * 100:.2f}%")

# Streamlit UI for user input and prediction
st.title("Disease Prediction Program oleh Fabian J Manoppo")
st.write("This program predicts either Hypertension or Heart Disease based on user input.")

# Select which disease to predict
disease_option = st.selectbox("Select Disease to Predict", ("Hypertension", "Heart Disease"))

# Input fields for Hypertension
if disease_option == "Hypertension":
    st.subheader("Enter Data for Hypertension Prediction")
    age = st.number_input("Age")
    systolic_bp = st.number_input("Systolic Blood Pressure")
    diastolic_bp = st.number_input("Diastolic Blood Pressure")
    cholesterol = st.number_input("Cholesterol")

    # Collect user input into a list and predict
    if st.button("Predict Hypertension"):
        input_data_hyp = [[age, systolic_bp, diastolic_bp, cholesterol]]  # Adjust based on dataset columns
        hyp_prediction = model_hyp.predict(input_data_hyp)
        if hyp_prediction[0] == 1:
            st.success("Prediction: Hypertension Detected")
        else:
            st.success("Prediction: No Hypertension Detected")

# Input fields for Heart Disease
elif disease_option == "Heart Disease":
    st.subheader("Enter Data for Heart Disease Prediction")
    age = st.number_input("Age")
    cholesterol = st.number_input("Cholesterol")
    resting_bp = st.number_input("Resting Blood Pressure")
    max_heart_rate = st.number_input("Max Heart Rate Achieved")
    
    # Collect user input into a list and predict
    if st.button("Predict Heart Disease"):
        input_data_hd = [[age, cholesterol, resting_bp, max_heart_rate]]  # Adjust based on dataset columns
        hd_prediction = model_hd.predict(input_data_hd)
        if hd_prediction[0] == 1:
            st.success("Prediction: Heart Disease Detected")
        else:
            st.success("Prediction: No Heart Disease Detected")
