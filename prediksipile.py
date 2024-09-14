import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load data from CSV file
@st.cache
def load_data():
    return pd.read_csv('datasetteori.csv')

df = load_data()

# Split data into features and target variable
X = df[['diameter', 'length', 'nspt1', 'nspt2', 'nspt3', 'qh', 'yh', 'bm', 'qa']]
y = df['sa']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Sidebar for user input
st.sidebar.header("Input parameters for prediction")
diameter = st.sidebar.number_input('Diameter (mm)', min_value=1.0, value=100.0)
length = st.sidebar.number_input('Length (m)', min_value=1.0, value=10.0)
nspt1 = st.sidebar.number_input('NSPT1', min_value=1.0, value=10.0)
nspt2 = st.sidebar.number_input('NSPT2', min_value=1.0, value=10.0)
nspt3 = st.sidebar.number_input('NSPT3', min_value=1.0, value=10.0)
qh = st.sidebar.number_input('Qh', min_value=0.0, value=50.0)
yh = st.sidebar.number_input('Yh', min_value=0.0, value=0.5)
bm = st.sidebar.number_input('Bm', min_value=0.0, value=0.1)
qa = st.sidebar.number_input('Qa', min_value=0.0, value=150.0)

# Create a DataFrame for the input values
input_data = pd.DataFrame({
    'diameter': [diameter],
    'length': [length],
    'nspt1': [nspt1],
    'nspt2': [nspt2],
    'nspt3': [nspt3],
    'qh': [qh],
    'yh': [yh],
    'bm': [bm],
    'qa': [qa]
})

# Prediction
prediction = model.predict(input_data)

# Display prediction
st.subheader('Predicted Settlement of Bore Pile (sa)')
st.write(prediction[0])

# Evaluate the model and display metrics
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader('Model Evaluation Metrics')
st.write(f'Mean Squared Error (MSE): {mse}')
st.write(f'Mean Absolute Error (MAE): {mae}')
st.write(f'R-squared (R²): {r2}')

# Feature Importance
st.subheader('Feature Importance')
feature_importances = model.feature_importances_
features = X.columns
fig, ax = plt.subplots()
ax.bar(features, feature_importances)
ax.set_xlabel('Features')
ax.set_ylabel('Importance')
ax.set_title('Feature Importance')
st.pyplot(fig)

# Plot Actual vs Predicted
st.subheader('Actual vs Predicted Settlement of Bore Pile (sa)')
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.set_xlabel('Actual Settlement of Bore Pile (sa)')
ax.set_ylabel('Predicted Settlement of Bore Pile (sa)')
ax.set_title('Actual vs Predicted Settlement of Bore Pile (sa)')

# Add diagonal line (y = x)
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')
ax.grid(True)
ax.legend()
st.pyplot(fig)
