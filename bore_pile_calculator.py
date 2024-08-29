import streamlit as st

def spt_method(N, diameter, depth):
    # Placeholder for actual SPT method calculation
    # N is the average SPT blow count, diameter in meters, depth in meters
    area = 3.14159 * (diameter / 2) ** 2
    capacity = N * area * depth  # Simplified example formula
    return capacity

def static_formula(unit_weight, diameter, depth):
    # Placeholder for actual Static Formula calculation
    # unit_weight in kN/m3, diameter in meters, depth in meters
    area = 3.14159 * (diameter / 2) ** 2
    capacity = unit_weight * area * depth  # Simplified example formula
    return capacity

# Streamlit interface
st.title('Bore Pile Foundation Load Capacity Calculator')

st.header('Input Parameters')
method = st.selectbox('Choose calculation method:', ['SPT Method', 'Static Formula'])
diameter = st.number_input('Diameter of the pile (m):', min_value=0.1, value=0.6, step=0.1)
depth = st.number_input('Depth of the pile (m):', min_value=1.0, value=6.0, step=1.0)

if method == 'SPT Method':
    N = st.number_input('Average SPT Blow Count (N):', min_value=10, max_value=100, value=50, step=5)
    if st.button('Calculate Capacity using SPT Method'):
        capacity = spt_method(N, diameter, depth)
        st.success(f'The estimated load capacity using SPT Method is {capacity:.2f} kN')
else:
    unit_weight = st.number_input('Soil Unit Weight (kN/m3):', min_value=10.0, max_value=30.0, value=18.0)
    if st.button('Calculate Capacity using Static Formula'):
        capacity = static_formula(unit_weight, diameter, depth)
        st.success(f'The estimated load capacity using Static Formula is {capacity:.2f} kN')
