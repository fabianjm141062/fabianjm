import streamlit as st

# Function to calculate vertical bearing capacity
def calculate_bearing_capacity(layers, diameter, pile_length, data_type, k=0.15, alpha=0.02):
    area_tip = 3.14159 * (diameter / 2) ** 2  # Pile tip area in square meters
    area_shaft = 3.14159 * diameter * pile_length  # Pile shaft area in square meters
    bearing_capacity = 0
    skin_friction_capacity = 0
    
    for layer in layers:
        if data_type == 'SPT Data':
            N_SPT = layer['N_SPT']
            end_bearing = N_SPT * 2 * area_tip * 10.1972  # Reduced SPT-based end bearing capacity
            skin_friction = k * N_SPT * area_shaft * 10.1972  # Reduced skin friction for SPT

        elif data_type == 'Laboratory Data':
            cu = layer['cu']
            end_bearing = 5 * cu * area_tip * 10.1972  # Reduced cu-based end bearing capacity
            skin_friction = alpha * cu * area_shaft * 10.1972  # Reduced cu-based skin friction

        elif data_type == 'Sondir Data':
            cone_resistance = layer['cone_resistance']
            end_bearing = 0.4 * cone_resistance * area_tip * 0.0981  # Reduced cone resistance-based end bearing capacity
            skin_friction = alpha * cone_resistance * area_shaft * 0.0981  # Reduced cone resistance-based skin friction

        bearing_capacity += end_bearing
        skin_friction_capacity += skin_friction

    total_capacity = bearing_capacity + skin_friction_capacity
    return total_capacity

# Function to calculate lateral bearing capacity for sandy clay soils
def calculate_lateral_bearing_capacity(diameter, pile_length, cu=None, N_SPT=None, cone_resistance=None, data_type="SPT Data"):
    if data_type == "SPT Data":
        Cl = 0.1 * N_SPT  # Reduced lateral factor for SPT
    elif data_type == "Laboratory Data":
        Cl = 0.2 * cu  # Reduced lateral factor for cu-based approach
    elif data_type == "Sondir Data":
        Cl = 0.1 * cone_resistance  # Reduced lateral factor for Sondir

    lateral_bearing_capacity = Cl * diameter * pile_length
    return lateral_bearing_capacity

# Function to calculate maximum bending moment based on lateral load and eccentricity
def calculate_bending_moment(P_l, pile_length, e_ratio=0.15):
    e = e_ratio * pile_length  # Reduced eccentricity (default is 0.15L)
    M_max = P_l * e  # Maximum bending moment
    return M_max

# Streamlit interface
st.title('Pile Foundation Bearing, Lateral Capacity & Bending Moment Calculator (Conservative Factors)')

# Select data type for input
data_type = st.selectbox('Select Data Type:', ['Laboratory Data', 'SPT Data', 'Sondir Data'])

# Input for soil layers based on selected data type
num_layers = st.number_input('Number of Soil Layers', min_value=1, max_value=5, value=3)
layers = []

# Input for specific data based on the data type selected
if data_type == 'Laboratory Data':
    st.header('Input Laboratory Data for Each Layer (Sandy Clay)')
    for i in range(num_layers):
        cu = st.number_input(f'Layer {i+1} - Undrained Shear Strength (cu) (kPa):', min_value=0, max_value=250, value=50)
        depth = st.number_input(f'Layer {i+1} - Layer Depth (m):', min_value=0.1, max_value=30.0, value=1.0)
        layers.append({'cu': cu, 'depth': depth})

elif data_type == 'SPT Data':
    st.header('Input SPT Data for Each Layer (Sandy Clay)')
    for i in range(num_layers):
        N_SPT = st.number_input(f'Layer {i+1} - N SPT Value:', min_value=0, max_value=60, value=10)
        depth = st.number_input(f'Layer {i+1} - Layer Depth (m):', min_value=0.1, max_value=30.0, value=1.0)
        layers.append({'N_SPT': N_SPT, 'depth': depth})

elif data_type == 'Sondir Data':
    st.header('Input Sondir Data for Each Layer (Sandy Clay)')
    for i in range(num_layers):
        cone_resistance = st.number_input(f'Layer {i+1} - Cone Resistance (kg/cm²):', min_value=0, max_value=250, value=100)
        depth = st.number_input(f'Layer {i+1} - Layer Depth (m):', min_value=0.1, max_value=30.0, value=1.0)
        layers.append({'cone_resistance': cone_resistance, 'depth': depth})

# Input for pile properties
diameter = st.number_input('Diameter of the pile (m):', min_value=0.1, value=0.6, step=0.1)
pile_length = st.number_input('Length of the pile (m):', min_value=1.0, value=6.0, step=1.0)

# Calculate bearing capacity, lateral capacity, and bending moment
if st.button('Calculate Bearing & Lateral Capacities'):
    capacity = calculate_bearing_capacity(layers, diameter, pile_length, data_type)
    
    if data_type == 'Laboratory Data':
        lateral_capacity = calculate_lateral_bearing_capacity(diameter, pile_length, cu=layers[0]['cu'], data_type="Laboratory Data")
    elif data_type == 'SPT Data':
        lateral_capacity = calculate_lateral_bearing_capacity(diameter, pile_length, N_SPT=layers[0]['N_SPT'], data_type="SPT Data")
    elif data_type == 'Sondir Data':
        lateral_capacity = calculate_lateral_bearing_capacity(diameter, pile_length, cone_resistance=layers[0]['cone_resistance'], data_type="Sondir Data")
    
    bending_moment = calculate_bending_moment(lateral_capacity, pile_length)
    
    st.success(f'The estimated load-bearing capacity of the pile is {capacity:.2f} tons')
    st.success(f'The estimated lateral bearing capacity of the pile is {lateral_capacity:.2f} tons')
    st.success(f'The estimated maximum bending moment is {bending_moment:.2f} ton-m')
