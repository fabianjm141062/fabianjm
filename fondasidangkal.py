import streamlit as st
import math

# Functions for bearing capacity calculation with unit conversions
def terzaghi_bearing_capacity(c, gamma, B, D_f, N_c, N_q, N_gamma):
    # Converting gamma from kg/m³ to kg/cm² (1 kN/m³ = 0.000102 kg/cm²)
    gamma = gamma * 0.000102
    return (c * N_c) + (gamma * D_f * N_q) + (0.5 * gamma * B * N_gamma)

def meyerhof_bearing_capacity(c, gamma, B, D_f, N_c, N_q, N_gamma, s_c, s_q, s_gamma):
    gamma = gamma * 0.000102
    return (c * N_c * s_c) + (gamma * D_f * N_q * s_q) + (0.5 * gamma * B * N_gamma * s_gamma)

# Bearing capacity factors (simplified example)
def bearing_capacity_factors(phi):
    N_c = (5.14 if phi == 0 else (1 + math.tan(math.radians(phi))) * (1 / math.tan(math.radians(45 + phi / 2))))
    N_q = (1 + math.tan(math.radians(phi))) * N_c
    N_gamma = (2 * (N_q - 1) * math.tan(math.radians(phi)))
    return N_c, N_q, N_gamma

# Streamlit app interface
st.title('Bearing Capacity Prediction - Shallow Foundation (Units: m, kg, kg/cm²) by Fabian J Manoppo')

# Method selection: Laboratory, Sondir, or SPT
data_source = st.selectbox('Select Data Source for Calculation', ['Laboratory Data', 'Sondir Data', 'SPT Data'])

# Method selection: Terzaghi or Meyerhof
calculation_method = st.selectbox('Select Calculation Method', ['Terzaghi', 'Meyerhof'])

# Input based on the selected data source
if data_source == 'Laboratory Data':
    st.subheader("Input Data from Laboratory Tests")
    phi = st.number_input("Angle of internal friction (degrees)", min_value=0.0, max_value=50.0, step=0.1)
    c = st.number_input("Cohesion (kg/cm²)", min_value=0.0, max_value=10.0, step=0.1)
    gamma = st.number_input("Unit weight of soil (kg/m³)", min_value=1000.0, max_value=25000.0, step=100.0)

elif data_source == 'Sondir Data':
    st.subheader("Input Data from Sondir (Cone Penetration Test)")
    qc = st.number_input("Cone resistance (Sondir, kg/cm²)", min_value=0.0, max_value=100.0, step=0.1)
    gamma = st.number_input("Unit weight of soil (kg/m³)", min_value=1000.0, max_value=25000.0, step=100.0)
    # Approximate method: Estimate cohesion and friction angle from qc
    # You may apply an empirical correlation here based on Sondir results
    c = qc / 10  # This is a simplification; actual formula depends on soil type
    phi = 30     # Assumed based on qc; can be updated with a better correlation

elif data_source == 'SPT Data':
    st.subheader("Input Data from SPT (Standard Penetration Test)")
    N_spt = st.number_input("SPT N-value", min_value=0, max_value=50, step=1)
    gamma = st.number_input("Unit weight of soil (kg/m³)", min_value=1000.0, max_value=25000.0, step=100.0)
    # Approximate method: Estimate cohesion and friction angle from SPT N-value
    # Use an empirical correlation
    c = N_spt / 5  # This is a simplification; use specific empirical equations
    phi = 28       # Assumed based on SPT N-value; can be updated with a better correlation

# Input for foundation dimensions
st.subheader("Foundation Dimensions")
B = st.number_input("Width of foundation (m)", min_value=0.5, max_value=10.0, step=0.1)
L = st.number_input("Length of foundation (m)", min_value=0.5, max_value=10.0, step=0.1)
D_f = st.number_input("Depth of foundation (m)", min_value=0.5, max_value=5.0, step=0.1)

# Submit button
if st.button("Calculate"):
    # Calculate bearing capacity factors based on the input friction angle (phi)
    N_c, N_q, N_gamma = bearing_capacity_factors(phi)

    # Calculate ultimate bearing capacity based on the selected method
    if calculation_method == 'Terzaghi':
        q_u = terzaghi_b
