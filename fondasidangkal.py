import streamlit as st
import math

# Functions for bearing capacity calculation with unit conversions
def terzaghi_bearing_capacity(c, gamma, B, D_f, N_c, N_q, N_gamma):
    # Converting gamma from kg/cm³ to kg/cm² (1 kg/cm³ = 10000 kg/m³ and 1 kg/m³ = 0.000102 kg/cm²)
    gamma = gamma * 0.000102 * 10000
    return (c * N_c) + (gamma * D_f * N_q) + (0.5 * gamma * B * N_gamma)

def meyerhof_bearing_capacity(c, gamma, B, D_f, N_c, N_q, N_gamma, s_c, s_q, s_gamma):
    # Converting gamma from kg/cm³ to kg/cm²
    gamma = gamma * 0.000102 * 10000
    return (c * N_c * s_c) + (gamma * D_f * N_q * s_q) + (0.5 * gamma * B * N_gamma * s_gamma)

# Bearing capacity factors (simplified example)
def bearing_capacity_factors(phi):
    N_c = (5.14 if phi == 0 else (1 + math.tan(math.radians(phi))) * (1 / math.tan(math.radians(45 + phi / 2))))
    N_q = (1 + math.tan(math.radians(phi))) * N_c
    N_gamma = (2 * (N_q - 1) * math.tan(math.radians(phi)))
    return N_c, N_q, N_gamma

# Streamlit app interface
st.title('Bearing Capacity Prediction - Shallow Foundation (Units: m, kg, kg/cm², kg/cm³) by Fabian J Manoppo')

# Method selection: Laboratory, Sondir, or SPT
data_source = st.selectbox('Select Data Source for Calculation', ['Laboratory Data', 'Sondir Data', 'SPT Data'])

# Method selection: Terzaghi or Meyerhof
calculation_method = st.selectbox('Select Calculation Method', ['Terzaghi', 'Meyerhof'])

# Input based on the selected data source
if data_source == 'Laboratory Data':
    st.subheader("Input Data from Laboratory Tests")
    phi = st.number_input("Angle of internal friction (degrees)", min_value=0.0, max_value=50.0, step=0.1)
    c = st.number_input("Cohesion (kg/cm²)", min_value=0.0, max_value=10.0, step=0.1)
    gamma = st.number_input("Unit weight of soil (kg/cm³)", min_value=0.001, max_value=0.025, step=0.001)

elif data_source == 'Sondir Data':
    st.subheader("Input Data from Sondir (Cone Penetration Test)")
    qc = st.number_input("Cone resistance (Sondir, kg/cm²)", min_value=0.0, max_value=100.0, step=0.1)
    gamma = st.number_input("Unit weight of soil (kg/cm³)", min_value=0.001, max_value=0.025, step=0.001)
    # Approximate method: Estimate cohesion and friction angle from qc
    c = qc / 10  # Simplification; actual formula depends on soil type
    phi = 30     # Assumed based on qc; can be updated with better correlation

elif data_source == 'SPT Data':
    st.subheader("Input Data from SPT (Standard Penetration Test)")
    N_spt = st.number_input("SPT N-value", min_value=0, max_value=50, step=1)
    gamma = st.number_input("Unit weight of soil (kg/cm³)", min_value=0.001, max_value=0.025, step=0.001)
    # Approximate method: Estimate cohesion and friction angle from SPT N-value
    c = N_spt / 5  # Simplification; use specific empirical equations
    phi = 28       # Assumed based on SPT N-value; can be updated with better correlation

# Input for foundation dimensions
st.subheader("Foundation Dimensions")
B = st.number_input("Width of foundation (m)", min_value=0.5, max_value=10.0, step=0.1)
L = st.number_input("Length of foundation (m)", min_value=0.5, max_value=10.0, step=0.1)
D_f = st.number_input("Depth of foundation (m)", min_value=0.5, max_value=5.0, step=0.1)

# Submit button to perform the calculation
if st.button("Calculate"):
    # Calculate bearing capacity factors based on the input friction angle (phi)
    N_c, N_q, N_gamma = bearing_capacity_factors(phi)

    # Calculate ultimate bearing capacity based on the selected method
    if calculation_method == 'Terzaghi':
        q_u = terzaghi_bearing_capacity(c, gamma, B, D_f, N_c, N_q, N_gamma)
        st.write(f"Ultimate Bearing Capacity (Terzaghi): {q_u:.2f} kg/cm²")
    elif calculation_method == 'Meyerhof':
        q_u = meyerhof_bearing_capacity(c, gamma, B, D_f, N_c, N_q, N_gamma, 1, 1, 1)
        st.write(f"Ultimate Bearing Capacity (Meyerhof): {q_u:.2f} kg/cm²")
