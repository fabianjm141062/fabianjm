import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats, optimize

# Function Definitions for Lateral Bearing Capacity

# 1. Manoppo and Koumoto Method
def manoppo_koumoto(Y, Q):
    Y_by_Q = np.array(Y) / np.array(Q)
    popt, _ = optimize.curve_fit(lambda Y, a, b: a + b * Y, Y, Y_by_Q)
    a, b = popt
    if b != 0:
        return 1 / b
    return None

# 2. Tangent Modulus Method
def tangent_modulus_method(Y, Q):
    slopes = np.gradient(Q, Y)
    threshold = 0.01  # Small threshold for near-zero slope
    for i, slope in enumerate(slopes):
        if abs(slope) < threshold:
            return Q[i]
    return Q[-1]

# 3. Chin-Kondner Extrapolation Method
def chin_method(Y, Q):
    Y_by_Q = np.array(Y) / np.array(Q)
    slope, intercept, _, _, _ = stats.linregress(Y, Y_by_Q)
    if intercept != 0:
        return 1 / intercept
    return None

# 4. Mazurkiwich Method (Quadratic Fitting)
def mazurkiwich_method(Y, Q):
    try:
        # Check if deflection and load arrays are non-empty and numeric
        if len(Y) < 3 or len(Q) < 3:
            return None  # Polyfit requires at least 3 points for a quadratic fit
        
        # Attempt a quadratic fit
        popt, _ = np.polyfit(Y, Q, 2, full=False)
        a, b, c = popt  # coefficients of the quadratic equation
        if a != 0:
            return -b / (2 * a)  # Maximum point of the quadratic equation (the ultimate capacity)
        else:
            return Q[-1]  # Return the last load if quadratic fitting doesn't work
    except Exception as e:
        # Handle any exception that occurs during polyfit
        print(f"Error in Mazurkiwich method: {str(e)}")
        return None

# Function to apply all methods
def apply_methods(Y, Q):
    results = {}
    results['Manoppo & Koumoto'] = manoppo_koumoto(Y, Q)
    results['Tangent Modulus'] = tangent_modulus_method(Y, Q)
    results['Chin Method'] = chin_method(Y, Q)
    results['Mazurkiwich Method'] = mazurkiwich_method(Y, Q)
    return results

# Streamlit Interface for Lateral Bearing Capacity Prediction
st.title("Lateral Bearing Capacity Prediction for pile loading test with machine learning AI data by Fabian J Manoppo")

st.write("This app predicts the lateral bearing capacity of vertical, positive, and negative batter piles using Manoppo & Koumoto,Tangent Modulus,Chin Method,Mazurkiwich Method  methods.")

# Input Section for Lateral Bearing Capacity
st.subheader("Input Pile Data (Lateral Loads)")

# Input: Deflection and Load data
deflection_data = st.text_area("Enter Deflection Data (comma-separated, mm)", "0, 5, 10, 15, 20, 25, 30, 35, 40")
load_data = st.text_area("Enter Lateral Load Data (comma-separated, kN)", "0.2, 0.18, 0.15, 0.12, 0.1, 0.08, 0.06, 0.04, 0.02")

# Convert input data to lists
deflection = [float(x) for x in deflection_data.split(",")]
load = [float(x) for x in load_data.split(",")]

# Predict button
if st.button("Predict Lateral Bearing Capacity"):
    if len(deflection) != len(load):
        st.error("Deflection and Load data must have the same length.")
    elif len(deflection) < 3:
        st.error("Need at least 3 data points for the Mazurkiwich Method.")
    else:
        # Apply the methods
        lateral_capacities = apply_methods(deflection, load)
        
        # Display the results
        st.subheader("Predicted Lateral Bearing Capacities:")
        for method, capacity in lateral_capacities.items():
            if capacity is not None:
                st.write(f"{method}: {capacity:.4f} kN")
            else:
                st.write(f"{method}: Could not compute")

# To run locally:
# 1. Save this code to a Python file (e.g., `streamlit_lateral_pile_app.py`)
# 2. Run the Streamlit app by typing in the terminal:
#    streamlit run streamlit_lateral_pile_app.py

