import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Streamlit configuration
st.title('Prediction of qa, sa, qh, yh, and bm using RandomForestRegressor oleh Fabian J Manoppo')

# Load dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Display first few rows of the dataset
    st.write("Dataset Preview:")
    st.write(df.head())

    # Define features (X) and targets (y)
    features = ['diameter', 'length', 'nspt1', 'nspt2', 'nspt3']  # Common features for all targets

    # Splitting data for each target
    X = df[features]
    
    targets = ['qa', 'sa', 'qh', 'yh', 'bm']
    predictions = {}
    metrics = {}

    for target in targets:
        y = df[target]
        
        # Splitting data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create RandomForestRegressor model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Store predictions and metrics
        predictions[target] = (y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics[target] = {
            "MSE": mse,
            "MAE": mae,
            "R2": r2
        }

    # Display the metrics for each target
    for target in targets:
        st.subheader(f'Metrics for {target}')
        st.write(f"Mean Squared Error (MSE): {metrics[target]['MSE']}")
        st.write(f"Mean Absolute Error (MAE): {metrics[target]['MAE']}")
        st.write(f"R-squared (R²): {metrics[target]['R2']}")
        
        # Plot Actual vs Predicted for each target
        y_test, y_pred = predictions[target]
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(y_test, y_pred)
        ax.set_xlabel(f'Actual {target}')
        ax.set_ylabel(f'Predicted {target}')
        ax.set_title(f'Actual vs Predicted {target}')

        # Diagonal line for perfect predictions
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

# Streamlit Run Command
# To run the app, use the following command in your terminal:
# streamlit run your_script_name.py
