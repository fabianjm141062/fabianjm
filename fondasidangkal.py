import streamlit as st
import math

# Functions for bearing capacity calculation with unit conversions
def terzaghi_bearing_capacity(c, gamma, B, D_f, N_c, N_q, N_gamma):
    # Converting gamma from kg/m³ to kg/cm² (1 kN/m³ = 0.000102 kg/cm²)
    gamma = gamma * 0.000102
    # Ultimate bearing capacity using Terzaghi
    return (c * N_c) + (gamma * D_f * N_q) + (0.5 * gamma * B * N_gamma)

def meyerhof_bearing_capacity(c, gamma, B, D_f, N_c, N_q, N_gamma, s_c, s_q, s_gamma):
    # Converting gamma from kg/m³ to kg/cm²
    gamma = gamma * 0.000102
    # Ultimate bearing capacity using Meyerhof
    return (c * N_c * s_c) + (gamma * D_f * N_q * s_q) + (0.5 * gamma * B * N_gamma * s_gamma)

# Bearing capacity factors (simplified example)
def bearing_capacity_factors(phi):
    N_c = (5.14 if phi == 0 else (1 + math.tan(math.radians(phi))) * (1 / math.tan(math.radians(45 + phi / 2))))
    N_q = (1 + math.tan(math.radians(phi))) * N_c
    N_gamma = (2 * (N_q - 1) * math.tan(math.radians(phi)))
    return N_c, N_q, N_gamma

# Streamlit app interface
st.title('Bearing Capacity Prediction - Shallow Foundation (Units: m, kg, kg/cm²)oleh Fabian J Manoppo')

# Input form for soil and foundation data
with st.form("input_form"):
    st.subheader("Soil Properties")
    phi = st.number_input("Angle of internal friction (degrees)", min_value=0.0, max_value=50.0, step=0.1)
    c = st.number_input("Cohesion (kg/cm²)", min_value=0.0, max_value=10.0, step=0.1)
    gamma = st.number_input("Unit weight of soil (kg/m³)", min_value=1000.0, max_value=25000.0, step=100.0)

    st.subheader("Foundation Dimensions")
    B = st.number_input("Width of foundation (m)", min_value=0.5, max_value=10.0, step=0.1)
    L = st.number_input("Length of foundation (m)", min_value=0.5, max_value=10.0, step=0.1)
    D_f = st.number_input("Depth of foundation (m)", min_value=0.5, max_value=5.0, step=0.1)

    st.subheader("Sondir and SPT Data")
    qc = st.number_input("Cone resistance (Sondir, kg/cm²)", min_value=0.0, max_value=100.0, step=0.1)
    N_spt = st.number_input("SPT N-value", min_value=0, max_value=50, step=1)

    # Submit button
    submitted = st.form_submit_button("Calculate")

    if submitted:
        # Calculate bearing capacity factors based on the input friction angle (phi)
        N_c, N_q, N_gamma = bearing_capacity_factors(phi)

        # Calculate ultimate bearing capacity using Terzaghi and Meyerhof methods
        terzaghi_q_u = terzaghi_bearing_capacity(c, gamma, B, D_f, N_c, N_q, N_gamma)
        meyerhof_q_u = meyerhof_bearing_capacity(c, gamma, B, D_f, N_c, N_q, N_gamma, 1, 1, 1)

        # Display results
        st.write(f"Ultimate Bearing Capacity (Terzaghi): {terzaghi_q_u:.2f} kg/cm²")
        st.write(f"Ultimate Bearing Capacity (Meyerhof): {meyerhof_q_u:.2f} kg/cm²")
