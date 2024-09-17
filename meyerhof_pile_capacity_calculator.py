import streamlit as st

def calculate_bearing_capacity_sandy_clay(layers, diameter, pile_length, k=0.15):
    """ Calculate bearing capacity for sandy clay using NSPT and laboratory values in tons. """
    area_tip = 3.14159 * (diameter / 2) ** 2  # Pile tip area in square meters
    area_shaft = 3.14159 * diameter * pile_length  # Pile shaft area in square meters
    bearing_capacity = 0
    skin_friction_capacity = 0
    
    for layer in layers:
        N_SPT = layer.get('N_SPT', 10)  # Default to NSPT of 10 if not given
        cu = layer.get('cu', 50)  # Default to cu of 50 kPa if not given
        
        # End Bearing Capacity based on NSPT and cu for sandy clay (in tons)
        end_bearing = (N_SPT * 3 + cu * 5) * area_tip * 10.1972  # Adjusted for sandy clay
        
        # Skin Friction (Shaft Resistance) - Mixed from NSPT and cu for sandy clay
        skin_friction = (k * N_SPT + 0.2 * cu) * area_shaft * 10.1972
        
        bearing_capacity += end_bearing
        skin_friction_capacity += skin_friction
    
    total_capacity = bearing_capacity + skin_friction_capacity
    return total_capacity

def calculate_lateral_bearing_capacity_sandy_clay(diameter, pile_length, cu=None, N_SPT=None):
    """ Calculate lateral bearing capacity for sandy clay soils. """
    
    # Lateral capacity factor for sandy clay, blending cu and NSPT-based approaches
    Cl = (0.3 * cu if cu else 0.15 * N_SPT)  # More conservative lateral factor for sandy clay
    
    lateral_bearing_capacity = Cl * diameter * pile_length  # Lateral bearing capacity
    return lateral_bearing_capacity

def calculate_bending_moment(P_l, pile_length, e_ratio=0.2):
    """ Calculate the maximum bending moment based on lateral load and eccentricity. """
    e = e_ratio * pile_length  # Eccentricity (default is 0.2L)
    M_max = P_l * e  # Maximum bending moment
    return M_max

# Streamlit application interface
st.title('Pile Foundation Bearing & Lateral Capacity & Moment (in Tons,Tons.m) for Sandy Clay Soils Theory Meyerhof by Fabian J Manoppo')

# Input for soil layers based on selected data type
num_layers = st.number_input('Number of Soil Layers', min_value=1, max_value=5, value=3)
layers = []

st.header('Input Sandy Clay Data for Each Layer')
for i in range(num_layers):
    st.write(f"### Layer {i+1}")
    N_SPT = st.number_input(f'Layer {i+1} - N SPT Value:', min_value=0, max_value=60, value=10)
    cu = st.number_input(f'Layer {i+1} - Undrained Shear Strength (cu) (kPa):', min_value=0, max_value=250, value=50)
    depth = st.number_input(f'Layer {i+1} - Layer Depth (m):', min_value=0.1, max_value=30.0, value=1.0)
    layers.append({'N_SPT': N_SPT, 'cu': cu, 'depth': depth})

# Input for pile properties
diameter = st.number_input('Diameter of the pile (m):', min_value=0.1, value=0.6, step=0.1)
pile_length = st.number_input('Length of the pile (m):', min_value=1.0, value=6.0, step=1.0)

# Calculate bearing capacity and lateral load
if st.button('Calculate Bearing & Lateral Capacities'):
    # Calculate bearing capacity for sandy clay soils
    capacity = calculate_bearing_capacity_sandy_clay(layers, diameter, pile_length)
    
    # Calculate lateral bearing capacity
    lateral_capacity = calculate_lateral_bearing_capacity_sandy_clay(diameter, pile_length, cu=layers[0]['cu'], N_SPT=layers[0]['N_SPT'])
    
    # Calculate bending moment
    bending_moment = calculate_bending_moment(lateral_capacity, pile_length)
    
    st.success(f'The estimated load-bearing capacity of the pile is {capacity:.2f} tons')
    st.success(f'The estimated lateral bearing capacity of the pile is {lateral_capacity:.2f} tons')
    st.success(f'The estimated maximum bending moment is {bending_moment:.2f} ton-m')
