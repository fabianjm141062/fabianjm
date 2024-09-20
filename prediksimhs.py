import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Function to load and process the dataset
@st.cache
def load_data():
    # Load the dataset (use your file path here)
    data = pd.read_csv('datamahasiswa.csv')
    return data

# Load dataset
st.title("Student Dropout and Study Duration Prediction App")
st.write("This app uses a Random Forest model to predict student dropout and study duration.")

data = load_data()

# Show the dataset
if st.checkbox('Show Dataset'):
    st.write(data)

# Add "tahun_lulus" as a feature by calculating it from the duration of study (for demo purposes)
# Assuming 'lama_studi' is years, we add it to the admission year to calculate "tahun_lulus"
data['tahun_lulus'] = data['tahun_masuk'].apply(lambda x: int(x.replace('TS-', ''))) + data['lama_studi']

# Data preparation
features = data.drop(columns=['no', 'prodi', 'tahun_masuk', 'do', 'lama_studi'])
dropout_target = data['do']
lama_studi_target = data['lama_studi']

# Splitting data into training and testing sets
X_train_do, X_test_do, y_train_do, y_test_do = train_test_split(features, dropout_target, test_size=0.2, random_state=42)
X_train_study, X_test_study, y_train_study, y_test_study = train_test_split(features, lama_studi_target, test_size=0.2, random_state=42)

# Train Random Forest Models for both targets
rf_model_do = RandomForestRegressor(random_state=42)
rf_model_do.fit(X_train_do, y_train_do)

rf_model_study = RandomForestRegressor(random_state=42)
rf_model_study.fit(X_train_study, y_train_study)

# Making predictions on the test set
y_pred_do = rf_model_do.predict(X_test_do)
y_pred_study = rf_model_study.predict(X_test_study)

# Display model performance (Mean Squared Error)
mse_do = mean_squared_error(y_test_do, y_pred_do)
mse_study = mean_squared_error(y_test_study, y_pred_study)

st.write(f"Dropout Prediction Mean Squared Error: {mse_do}")
st.write(f"Study Duration Prediction Mean Squared Error: {mse_study}")

# User input for new predictions
st.header("Predict Dropout and Study Duration for a New Data Input")

# Collecting input from the user
input_data = {}
for col in features.columns:
    input_data[col] = st.number_input(f'Enter {col}', min_value=0)

# Additional input for "tahun_lulus" (Graduation Year)
input_data['tahun_lulus'] = st.number_input('Enter Graduation Year (tahun_lulus)', min_value=2000, max_value=2100)

# Convert input to DataFrame
input_df = pd.DataFrame([input_data])

# Prediction on new input
if st.button('Predict'):
    pred_do = rf_model_do.predict(input_df)
    pred_study = rf_model_study.predict(input_df)
    
    st.write(f'Predicted dropout: {pred_do[0]}')
    st.write(f'Predicted duration of study (years): {pred_study[0]}')

# Optionally: Display input data
if st.checkbox('Show Input Data'):
    st.write(input_df)
