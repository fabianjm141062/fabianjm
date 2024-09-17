import streamlit as st

def calculate_bearing_capacity_nspt(layers, diameter, pile_length, k=0.2):
    """ Calculate bearing capacity for sand using NSPT values in tons. """
    area_tip = 3.14159 * (diameter / 2) ** 2  # Pile tip area in square meters
    area_shaft = 3.14159 * diameter * pile_length  # Pile shaft area in square meters
    bearing_capacity = 0
    skin_friction_capacity = 0
    
    for layer in layers:
        N_SPT = layer['N_SPT']
        
        # End Bearing Capacity based on NSPT (in tons)
        end_bearing = N_SPT * 4 * area_tip * 10.1972  # Adjusted NSPT formula for sands
        
        # Skin Friction (Shaft Resistance)
        skin_friction = k * N_SPT * area_shaft * 10.1972  # Adjusted skin friction calculation
        
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
        skin_friction = alpha * cone_resistance * area_shaft * 0.0981  # Empirical factor alpha for friction
        
        bearing_capacity += end_bearing
        skin_friction_capacity += skin_friction
    
    total_capacity = bearing_capacity + skin_friction_capacity
    return total_capacity

def calculate_bearing_capacity_lab(layers, diameter, pile_length, alpha=0.5):
    """ Calculate bearing capacity for clay (Laboratory Data) using Meyerhof's formula. """
    area_tip = 3.14159 * (diameter / 2) ** 2  # Pile tip area in square meters
    area_shaft = 3.14159 * diameter * pile_length  # Pile shaft area in square meters
    bearing_capacity = 0
    skin_friction_capacity = 0
    
    for layer in layers:
        cu = layer['cu']
        
        # End Bearing Capacity for clay using Meyerhof's formula (in tons)
        end_bearing = 9 * cu * area_tip * 10.1972  # Convert kPa to tons
        
        # Skin Friction (Shaft Resistance)
        skin_friction = alpha * cu * area_shaft * 10.1972  # Simplified skin friction for clay
        
        bearing_capacity += end_bearing
        skin_friction_capacity += skin_friction
    
    total_capacity = bearing_capacity + skin_friction_capacity
    return total_capacity

def calculate_lateral_bearing_capacity(diameter, pile_length, cu=None, N_SPT=None, cone_resistance=None, soil_type="clay"):
    """ Calculate lateral bearing capacity for both cohesive (clay) and granular (sand) soils. """
    
    # Lateral capacity factor
    if soil_type == "clay":
        Cl = 0.5 * cu  # for clay
    elif soil_type == "sand":
        Cl = 0.33 * N_SPT if N_SPT else 0.6 * cone_resistance  # for sand
    
    lateral_bearing_capacity = Cl * diameter * pile_length  # Lateral bearing capacity
    return lateral_bearing_capacity

def calculate_bending_moment(P_l, pile_length, e_ratio=0.2):
    """ Calculate the maximum bending moment based on lateral load and eccentricity. """
    e = e_ratio * pile_length  # Eccentricity (default is 0.2L)
    M_max = P_l * e  # Maximum bending moment
    return M_max

# Streamlit application interface
st.title('Driven Pile Foundation Bearing, Lateral Capacity & Moment Meyerhof Theory oleh Fabian J Manoppo (in Tons & Tons.m)')

# Select type of data for input
data_type = st.selectbox('Select Data Type:', ['NSPT Data (Sandy Soil)', 'Dutch Cone Penetrometer Data (Sandy Soil)', 'Laboratory Data (Clay Soil)'])

# Input for soil layers based on selected data type
num_layers = st.number_input('Number of Soil Layers', min_value=1, max_value=5, value=3)
layers = []

if data_type == 'NSPT Data (Sandy Soil)':
    st.header('Input NSPT Data for Each Layer')
    for i in range(num_layers):
        st.write(f"### Layer {i+1}")
        N_SPT = st.number_input(f'Layer {i+1} - N SPT Value:', min_value=0, max_value=60, value=10)
        depth = st.number_input(f'Layer {i+1} - Layer Depth (m):', min_value=0.1, max_value=30.0, value=1.0)
        layers.append({'N_SPT': N_SPT, 'depth': depth})

elif data_type == 'Dutch Cone Penetrometer Data (Sandy Soil)':
    st.header('Input Dutch Cone Penetrometer Data for Each Layer')
    for i in range(num_layers):
        st.write(f"### Layer {i+1}")
        cone_resistance = st.number_input(f'Layer {i+1} - Cone Resistance (kg/cm²):', min_value=0, max_value=250, value=100)
        depth = st.number_input(f'Layer {i+1} - Layer Depth (m):', min_value=0.1, max_value=30.0, value=1.0)
        layers.append({'cone_resistance': cone_resistance, 'depth': depth})

elif data_type == 'Laboratory Data (Clay Soil)':
    st.header('Input Clay Data for Each Layer')
    for i in range(num_layers):
        st.write(f"### Layer {i+1}")
        cu = st.number_input(f'Layer {i+1} - Undrained Shear Strength (cu) (kPa):', min_value=0, max_value=250, value=100)
        depth = st.number_input(f'Layer {i+1} - Layer Depth (m):', min_value=0.1, max_value=30.0, value=1.0)
        layers.append({'cu': cu, 'depth': depth})

# Input for pile properties
diameter = st.number_input('Diameter of the pile (m):', min_value=0.1, value=0.6, step=0.1)
pile_length = st.number_input('Length of the pile (m):', min_value=1.0, value=6.0, step=1.0)

# Calculate bearing capacity and lateral load
if st.button('Calculate Bearing & Lateral Capacities'):
    if data_type == 'NSPT Data (Sandy Soil)':
        capacity = calculate_bearing_capacity_nspt(layers, diameter, pile_length)
        lateral_capacity = calculate_lateral_bearing_capacity(diameter, pile_length, N_SPT=layers[0]['N_SPT'], soil_type="sand")
    elif data_type == 'Dutch Cone Penetrometer Data (Sandy Soil)':
        capacity = calculate_bearing_capacity_cone(layers, diameter, pile_length)
        lateral_capacity = calculate_lateral_bearing_capacity(diameter, pile_length, cone_resistance=layers[0]['cone_resistance'], soil_type="sand")
    elif data_type == 'Laboratory Data (Clay Soil)':
        capacity = calculate_bearing_capacity_lab(layers, diameter, pile_length)
        lateral_capacity = calculate_lateral_bearing_capacity(diameter, pile_length, cu=layers[0]['cu'], soil_type="clay")
    
    # Calculate bending moment
    bending_moment = calculate_bending_moment(lateral_capacity, pile_length)
    
    st.success(f'The estimated load-bearing capacity of the pile is {capacity:.2f} tons')
    st.success(f'The estimated lateral bearing capacity of the pile is {lateral_capacity:.2f} tons')
    st.success(f'The estimated maximum bending moment is {bending_moment:.2f} ton-m')
