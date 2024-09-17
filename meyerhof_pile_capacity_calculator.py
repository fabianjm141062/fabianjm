import streamlit as st

def calculate_bearing_capacity_lab(layers, diameter, pile_length):
    """ Calculate bearing capacity based on laboratory test data in tons using Meyerhof's formula. """
    area = 3.14159 * (diameter / 2) ** 2
    bearing_capacity = 0
    
    for layer in layers:
        cohesion, unit_weight, depth = layer['cohesion'], layer['unit_weight'], layer['depth']
        
        # Meyerhof's calculations for laboratory data
        # Convert kPa to MPa (1 MPa = 10.1972 tons/m²)
        end_bearing = cohesion * 9 * area * 10.1972  # End bearing in tons
        skin_friction = 0.1 * unit_weight * area * depth * 10.1972  # Skin friction in tons
        
        layer_capacity = end_bearing + skin_friction
        bearing_capacity += layer_capacity
    
    return bearing_capacity

def calculate_bearing_capacity_nspt(layers, diameter, pile_length):
    """ Calculate bearing capacity based on NSPT values in tons. """
    area = 3.14159 * (diameter / 2) ** 2
    bearing_capacity = 0
    
    for layer in layers:
        N_SPT, depth = layer['N_SPT'], layer['depth']
        
        # End bearing in tons
        end_bearing = N_SPT * 40 * area * 10.1972
        skin_friction = 0  # No skin friction calculation without unit weight
        
        layer_capacity = end_bearing + skin_friction
        bearing_capacity += layer_capacity
    
    return bearing_capacity

def calculate_bearing_capacity_cone(layers, diameter, pile_length):
    """ Calculate bearing capacity based on Dutch Cone Penetrometer data in tons. """
    area = 3.14159 * (diameter / 2) ** 2
    bearing_capacity = 0
    
    for layer in layers:
        cone_resistance, depth = layer['cone_resistance'], layer['depth']
        
        # End bearing in tons
        end_bearing = cone_resistance * area * 10.1972  # Convert MPa to tons/m²
        skin_friction = 0  # No skin friction calculation without unit weight
        
        layer_capacity = end_bearing + skin_friction
        bearing_capacity += layer_capacity
    
    return bearing_capacity

# Streamlit application interface
st.title('Pile Foundation Bearing Capacity Calculator (in Tons)')

# Select type of data for input
data_type = st.selectbox('Select Data Type:', ['Laboratory Data', 'NSPT Data', 'Dutch Cone Penetrometer Data'])

# Input for soil layers based on selected data type
num_layers = st.number_input('Number of Soil Layers', min_value=1, max_value=5, value=3)
layers = []

if data_type == 'Laboratory Data':
    st.header('Input Laboratory Data for Each Layer')
    for i in range(num_layers):
        st.write(f"### Layer {i+1}")
        cohesion = st.number_input(f'Layer {i+1} - Cohesion (kPa):', min_value=0, max_value=100, value=0)
        unit_weight = st.number_input(f'Layer {i+1} - Unit Weight of Soil (kN/m³):', min_value=10.0, max_value=25.0, value=18.0)
        depth = st.number_input(f'Layer {i+1} - Layer Depth (m):', min_value=0.1, max_value=30.0, value=1.0)
        layers.append({'cohesion': cohesion, 'unit_weight': unit_weight, 'depth': depth})

elif data_type == 'NSPT Data':
    st.header('Input NSPT Data for Each Layer')
    for i in range(num_layers):
        st.write(f"### Layer {i+1}")
        N_SPT = st.number_input(f'Layer {i+1} - N SPT Value:', min_value=0, max_value=100, value=10)
        depth = st.number_input(f'Layer {i+1} - Layer Depth (m):', min_value=0.1, max_value=30.0, value=1.0)
        layers.append({'N_SPT': N_SPT, 'depth': depth})

elif data_type == 'Dutch Cone Penetrometer Data':
    st.header('Input Dutch Cone Penetrometer Data for Each Layer')
    for i in range(num_layers):
        st.write(f"### Layer {i+1}")
        cone_resistance = st.number_input(f'Layer {i+1} - Cone Resistance (MPa):', min_value=0.1, max_value=50.0, value=10.0)
        depth = st.number_input(f'Layer {i+1} - Layer Depth (m):', min_value=0.1, max_value=30.0, value=1.0)
        layers.append({'cone_resistance': cone_resistance, 'depth': depth})

# Input for pile properties
diameter = st.number_input('Diameter of the pile (m):', min_value=0.1, value=0.6, step=0.1)
pile_length = st.number_input('Length of the pile (m):', min_value=1.0, value=6.0, step=1.0)

# Calculate bearing capacity based on selected data type
if st.button('Calculate Bearing Capacity'):
    if data_type == 'Laboratory Data':
        capacity = calculate_bearing_capacity_lab(layers, diameter, pile_length)
    elif data_type == 'NSPT Data':
        capacity = calculate_bearing_capacity_nspt(layers, diameter, pile_length)
    elif data_type == 'Dutch Cone Penetrometer Data':
        capacity = calculate_bearing_capacity_cone(layers, diameter, pile_length)
    
    st.success(f'The estimated load-bearing capacity of the pile is {capacity:.2f} tons')
