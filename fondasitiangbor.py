import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Title of the app
st.title('Prediction of Working Vertical Load (Qa), Settlement (Sa), Lateral Capacity (Qh), Lateral Deflection (yh), and Bending Moment (bm) using Machine Learning by fabian J Manoppo')

# Load dataset directly from the file
df = pd.read_csv('datasetteori.csv')

# Display first few rows of the dataset
st.write("Dataset Preview:")
st.write(df.head())

# Define features (X) and targets (y)
X = df[['diameter', 'length', 'nspt1', 'nspt2', 'nspt3']]  # Input features
y = df[['qa', 'sa', 'qh', 'yh', 'bm']]  # Target variables

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create RandomForestRegressor model and wrap it in MultiOutputRegressor
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# Create input fields for manual input in Streamlit with units
st.subheader("Input values for prediction (with units):")
diameter = st.number_input("Diameter (m):", value=0.0)
length = st.number_input("Length (m):", value=0.0)
nspt1 = st.number_input("NSPT1 (blows/ft):", value=0.0)
nspt2 = st.number_input("NSPT2 (blows/ft):", value=0.0)
nspt3 = st.number_input("NSPT3 (blows/ft):", value=0.0)

# When the button is clicked, perform prediction
if st.button("Predict"):
    # Create a NumPy array of the manually input features
    input_data = np.array([[diameter, length, nspt1, nspt2, nspt3]])

    # Predict the target values based on the input
    predicted_values = model.predict(input_data)

    # Display the predicted results with units
    predicted_qa, predicted_sa, predicted_qh, predicted_yh, predicted_bm = predicted_values[0]
    st.write("Predicted values based on your inputs (with units):")
    st.write(f"Predicted qa (kN/m²): {predicted_qa:.2f}")
    st.write(f"Predicted sa (mm): {predicted_sa:.2f}")
    st.write(f"Predicted qh (kN): {predicted_qh:.2f}")
    st.write(f"Predicted yh (mm): {predicted_yh:.2f}")
    st.write(f"Predicted bm (kNm): {predicted_bm:.2f}")

# Optionally, evaluate the model on test data and show performance metrics
y_pred_test = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_test, multioutput='raw_values')
mae = mean_absolute_error(y_test, y_pred_test, multioutput='raw_values')
r2 = r2_score(y_test, y_pred_test, multioutput='raw_values')

st.subheader("Model performance on test set (with units):")
st.write(f"Mean Squared Error (MSE) for qa, sa, qh, yh, bm: {mse}")
st.write(f"Mean Absolute Error (MAE) for qa, sa, qh, yh, bm: {mae}")
st.write(f"R-squared (R²) for qa, sa, qh, yh, bm: {r2}")
