import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

def simulate_column_removal(soil_modulus, rockfill_modulus, initial_stress, depth):
    """
    Simulate the stress distribution in a rockfill column during casing removal.
    This is a placeholder function. Replace with actual numerical method.
    """
    x = np.linspace(0, depth, 100)
    stress = initial_stress * np.exp(-x / soil_modulus) + rockfill_modulus * np.log(x + 1)
    return x, stress

st.title("Numerical Model Analysis of Cassings Effect Rockfill Column Construction by Fabian J Manoppo")

st.sidebar.header("Input Parameters")
soil_modulus = st.sidebar.number_input("Soil Modulus (kPa)", min_value=1000, max_value=50000, value=10000)
rockfill_modulus = st.sidebar.number_input("Rockfill Modulus (kPa)", min_value=1000, max_value=50000, value=15000)
initial_stress = st.sidebar.number_input("Initial Stress (kPa)", min_value=100, max_value=1000, value=500)
depth = st.sidebar.number_input("Depth of Column (m)", min_value=1, max_value=100, value=10)

if st.sidebar.button("Simulate Removal"):
    x, stress = simulate_column_removal(soil_modulus, rockfill_modulus, initial_stress, depth)
    
    fig, ax = plt.subplots()
    ax.plot(x, stress)
    ax.set_xlabel("Depth (m)")
    ax.set_ylabel("Stress (kPa)")
    ax.set_title("Stress Distribution in Rockfill Column During Casing Removal")
    st.pyplot(fig)

