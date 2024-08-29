import pandas as pd
import streamlit as st

def calculate_bearing_capacity(layers, diameter):
    """ Calculate bearing capacity based on soil layers and pile diameter. """
    area = 3.14159 * (diameter / 2) ** 2
    bearing_capacity = 0
    
    for layer in layers:
        N_SPT, cohesion, depth = layer['N_SPT'], layer['cohesion'], layer['depth']
        if cohesion > 0:  # Cohesive soil
            capacity = cohesion * 9 * area  # Simplified method, 9 is an assumed bearing capacity factor
        else:  # Granular soil
            capacity = N_SPT * 0.1 * area * depth  # Simplified method, 0.1 is an assumed empirical factor
        
        bearing_capacity += capacity
    
    return bearing_capacity

# Streamlit application interface
st.title('Pile Foundation Bearing Capacity Calculator')

# Input for soil layers
num_layers = st.number_input('Number of Soil Layers', min_value=1, max_value=5, value=3)
layers = []
for i in range(num_layers):
    st.write(f"### Layer {i+1}")
    N_SPT = st.number_input(f'Layer {i+1} - N SPT Value:', min_value=0, max_value=100, value=10)
    cohesion = st.number_input(f'Layer {i+1} - Cohesion (kPa):', min_value=0, max_value=100, value=0)
    depth = st.number_input(f'Layer {i+1} - Layer Depth (m):', min_value=0.1, max_value=30.0, value=1.0)
    layers.append({'N_SPT': N_SPT, 'cohesion': cohesion, 'depth': depth})

diameter = st.number_input('Diameter of the pile (m):', min_value=0.1, value=0.6, step=0.1)

if st.button('Calculate Bearing Capacity'):
    capacity = calculate_bearing_capacity(layers, diameter)
    st.success(f'The estimated load-bearing capacity of the pile is {capacity:.2f} kN')


