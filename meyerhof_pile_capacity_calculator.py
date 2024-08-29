import streamlit as st

def calculate_bearing_capacity(layers, diameter, pile_length):
    """ Calculate bearing capacity based on soil layers, pile diameter, and pile length using Meyerhof's formula. """
    area = 3.14159 * (diameter / 2) ** 2
    bearing_capacity = 0
    
    for layer in layers:
        N_SPT, cohesion, unit_weight, depth = layer['N_SPT'], layer['cohesion'], layer['unit_weight'], layer['depth']
        
        # Meyerhof's calculations
        end_bearing = cohesion * 9 * area  # Simplified, typically involves factors based on phi
        skin_friction = 0.1 * unit_weight * area * depth  # Simplified method, often involves log terms and phi
        
        layer_capacity = end_bearing + skin_friction
        bearing_capacity += layer_capacity
    
    return bearing_capacity

# Streamlit application interface
st.title('Pile Foundation Bearing Capacity Calculator using Meyerhof Method')

# Input for soil layers
num_layers = st.number_input('Number of Soil Layers', min_value=1, max_value=5, value=3)
layers = []
for i in range(num_layers):
    st.write(f"### Layer {i+1}")
    cohesion = st.number_input(f'Layer {i+1} - Cohesion (kPa):', min_value=0, max_value=100, value=0)
    unit_weight = st.number_input(f'Layer {i+1} - Unit Weight of Soil (kN/m³):', min_value=10.0, max_value=25.0, value=18.0)
    depth = st.number_input(f'Layer {i+1} - Layer Depth (m):', min_value=0.1, max_value=30.0, value=1.0)
    N_SPT = st.number_input(f'Layer {i+1} - N SPT Value:', min_value=0, max_value=100, value=10)
    layers.append({'N_SPT': N_SPT, 'cohesion': cohesion, 'unit_weight': unit_weight, 'depth': depth})

diameter = st.number_input('Diameter of the pile (m):', min_value=0.1, value=0.6, step=0.1)
pile_length = st.number_input('Length of the pile (m):', min_value=1.0, value=6.0, step=1.0)

if st.button('Calculate Bearing Capacity'):
    capacity = calculate_bearing_capacity(layers, diameter, pile_length)
    st.success(f'The estimated load-bearing capacity of the pile is {capacity:.2f} kN')
