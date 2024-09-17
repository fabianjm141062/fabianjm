import streamlit as st

def calculate_bearing_capacity_nspt(layers, diameter, pile_length, k=0.7):
    """ Calculate bearing capacity for sand using NSPT values in tons. """
    area_tip = 3.14159 * (diameter / 2) ** 2  # Pile tip area in square meters
    area_shaft = 3.14159 * diameter * pile_length  # Pile shaft area in square meters
    bearing_capacity = 0
    skin_friction_capacity = 0
    
    for layer in layers:
        N_SPT = layer['N_SPT']
        
        # End Bearing Capacity based on NSPT (in tons)
        end_bearing = N_SPT * 40 * area_tip * 10.1972  # Convert NSPT to tons
        
        # Skin Friction (Shaft Resistance)
        skin_friction = k * N_SPT * area_shaft * 10.1972  # Empirical coefficient k * NSPT
        
        bearing_capacity += end_bearing
        skin_friction_capacity += skin_friction
    
    total_capacity = bearing_capacity + skin_friction_capacity
    return total_capacity

def calculate_bearing_capacity_cone(layers, diameter, pile_length, alpha=0.03):
    """ Calculate bearing capacity based on Dutch Cone Penetrometer data in tons. """
    area_tip = 3.14159 * (diameter / 2) ** 2  # Pile tip area in square meters
    area_shaft = 3.14159 * diameter * pile_length  # Pile shaft area in square meters
    bearing_capacity = 0
    skin_friction_capacity = 0
    
    for layer in layers:
        cone_resistance = layer['cone_resistance']
        
        # End Bearing Capacity (in tons) based on Cone Penetrometer data
        end_bearing = 0.6 * cone_resistance * area_tip * 0.0981  # Convert kg/cm² to tons/m²
        
        # Skin Friction (Shaft Resistance)
        skin_friction = alpha * cone_resistance * area_shaft * 0.0981  # Empirical factor alpha
        
        bearing_capacity += end_bearing
        skin_friction_capacity += skin_friction
    
    total_capacity = bearing_capacity + skin_friction_capacity
    return total_capacity

# Streamlit application interface
st.title('Pile Foundation Bearing Capacity Calculator with Meyerhof Theory by Fabian J Manoppo (in Tons)')

# Select type of data for input
data_type = st.selectbox('Select Data Type:', ['NSPT Data', 'Dutch Cone Penetrometer Data'])

# Input for soil layers based on selected data type
num_layers = st.number_input('Number of Soil Layers', min_value=1, max_value=5, value=3)
layers = []

if data_type == 'NSPT Data':
    st.header('Input NSPT Data for Each Layer')
    for i in range(num_layers):
        st.write(f"### Layer {i+1}")
        N_SPT = st.number_input(f'Layer {i+1} - N SPT Value:', min_value=0, max_value=60, value=10)
        depth = st.number_input(f'Layer {i+1} - Layer Depth (m):', min_value=0.1, max_value=30.0, value=1.0)
        layers.append({'N_SPT': N_SPT, 'depth': depth})

elif data_type == 'Dutch Cone Penetrometer Data':
    st.header('Input Dutch Cone Penetrometer Data for Each Layer')
    for i in range(num_layers):
        st.write(f"### Layer {i+1}")
        cone_resistance = st.number_input(f'Layer {i+1} - Cone Resistance (kg/cm²):', min_value=0, max_value=250, value=100)
        depth = st.number_input(f'Layer {i+1} - Layer Depth (m):', min_value=0.1, max_value=30.0, value=1.0)
        layers.append({'cone_resistance': cone_resistance, 'depth': depth})

# Input for pile properties
diameter = st.number_input('Diameter of the pile (m):', min_value=0.1, value=0.6, step=0.1)
pile_length = st.number_input('Length of the pile (m):', min_value=1.0, value=6.0, step=1.0)

# Calculate bearing capacity based on selected data type
if st.button('Calculate Bearing Capacity'):
    if data_type == 'NSPT Data':
        capacity = calculate_bearing_capacity_nspt(layers, diameter, pile_length)
    elif data_type == 'Dutch Cone Penetrometer Data':
        capacity = calculate_bearing_capacity_cone(layers, diameter, pile_length)
    
    st.success(f'The estimated load-bearing capacity of the pile is {capacity:.2f} tons')
