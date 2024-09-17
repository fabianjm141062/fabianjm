import pandas as pd

# URLs for the datasets
hypertension_url = "URL_OF_HYPERTENSION_DATASET"  # Replace with the actual URL
heart_disease_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

# Load the datasets
hypertension_data = pd.read_csv(hypertension_url)
heart_disease_data = pd.read_csv(heart_disease_url, header=None)

# Display first few rows of each dataset
print("Hypertension Dataset:")
print(hypertension_data.head())
print("\nHeart Disease Dataset:")
print(heart_disease_data.head())
# Preprocessing for Hypertension Dataset
# (Assuming 'Target' is the column that indicates Hypertension)
hypertension_data.fillna(hypertension_data.mean(), inplace=True)

# Preprocessing for Heart Disease Dataset (handling missing values, renaming columns)
heart_disease_data.columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"]
heart_disease_data.fillna(heart_disease_data.mean(), inplace=True)

# Convert categorical columns to numerical if necessary using one-hot encoding
hypertension_data = pd.get_dummies(hypertension_data)
heart_disease_data = pd.get_dummies(heart_disease_data)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Splitting Hypertension data
X_hyp = hypertension_data.drop('Target', axis=1)  # Assuming 'Target' column is for Hypertension prediction
y_hyp = hypertension_data['Target']
X_hyp_train, X_hyp_test, y_hyp_train, y_hyp_test = train_test_split(X_hyp, y_hyp, test_size=0.2, random_state=42)

# Splitting Heart Disease data
X_hd = heart_disease_data.drop('target', axis=1)
y_hd = heart_disease_data['target']
X_hd_train, X_hd_test, y_hd_train, y_hd_test = train_test_split(X_hd, y_hd, test_size=0.2, random_state=42)

# Train Random Forest models
model_hyp = RandomForestClassifier()
model_hd = RandomForestClassifier()

model_hyp.fit(X_hyp_train, y_hyp_train)
model_hd.fit(X_hd_train, y_hd_train)

# Predict and evaluate both models
hyp_pred = model_hyp.predict(X_hyp_test)
hd_pred = model_hd.predict(X_hd_test)

hyp_accuracy = accuracy_score(y_hyp_test, hyp_pred)
hd_accuracy = accuracy_score(y_hd_test, hd_pred)

print(f"Hypertension Prediction Model Accuracy: {hyp_accuracy * 100:.2f}%")
print(f"Heart Disease Prediction Model Accuracy: {hd_accuracy * 100:.2f}%")
import streamlit as st

# Streamlit user interface for disease prediction
st.title("Disease Prediction Program")
disease_option = st.selectbox("Select Disease to Predict", ("Hypertension", "Heart Disease"))

# Input fields for user data based on selected disease
if disease_option == "Hypertension":
    st.subheader("Enter Data for Hypertension Prediction")
    age = st.number_input("Age")
    systolic_bp = st.number_input("Systolic Blood Pressure")
    diastolic_bp = st.number_input("Diastolic Blood Pressure")
    cholesterol = st.number_input("Cholesterol")

    if st.button("Predict Hypertension"):
        input_data_hyp = [[age, systolic_bp, diastolic_bp, cholesterol]]  # Adjust input based on your dataset
        prediction_hyp = model_hyp.predict(input_data_hyp)
        if prediction_hyp[0] == 1:
            st.success("Prediction: Hypertension Detected")
        else:
            st.success("Prediction: No Hypertension Detected")

elif disease_option == "Heart Disease":
    st.subheader("Enter Data for Heart Disease Prediction")
    age = st.number_input("Age")
    cholesterol = st.number_input("Cholesterol")
    resting_bp = st.number_input("Resting Blood Pressure")
    max_heart_rate = st.number_input("Max Heart Rate Achieved")
    
    if st.button("Predict Heart Disease"):
        input_data_hd = [[age, cholesterol, resting_bp, max_heart_rate]]  # Adjust input based on your dataset
        prediction_hd = model_hd.predict(input_data_hd)
        if prediction_hd[0] == 1:
            st.success("Prediction: Heart Disease Detected")
        else:
            st.success("Prediction: No Heart Disease Detected")
