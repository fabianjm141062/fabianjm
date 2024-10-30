import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# Default properties for materials and soils
material_properties = {
    "steel": {"Density": 7850, "Yield Strength": 250, "Modulus of Elasticity": 200000},
    "concrete": {"Density": 2400, "Compressive Strength": 30, "Modulus of Elasticity": 30000}
}

soil_properties = {
    "sand": {"Unit Weight": 18, "Friction Angle": 35, "Cohesion": 0},
    "clay": {"Unit Weight": 20, "Friction Angle": 0, "Cohesion": 25}  # Example properties
}

class SheetPileAnalysis:
    def __init__(self, material, soil_layers, passive_layer, surcharge, groundwater_level):
        self.material = material
        self.soil_layers = soil_layers
        self.passive_layer = passive_layer
        self.surcharge = surcharge  # Uniform surcharge load (kN/m²)
        self.groundwater_level = groundwater_level  # Depth of groundwater level from top (m)
        self.total_depth = sum(layer['Depth'] for layer in soil_layers)  # Total depth of active soil

    def calculate_earth_pressure_coefficient(self, friction_angle, is_passive=False):
        if is_passive:
            K = np.tan(np.radians(45 + friction_angle / 2)) ** 2
        else:
            K = np.tan(np.radians(45 - friction_angle / 2)) ** 2
        return K

    def plot_pressure_diagram(self):
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.invert_yaxis()
        
        # Active Pressure Calculation with surcharge and groundwater effect
        depths = [0]
        active_pressures = [0]
        y_offset = 0
        
        for layer in self.soil_layers:
            Ka = self.calculate_earth_pressure_coefficient(layer['Friction Angle'])
            gamma = layer['Unit Weight']
            height = layer['Depth']
            
            # Adjust for groundwater level if below the layer
            if y_offset + height > self.groundwater_level:
                gamma -= 9.81  # Adjust unit weight for submerged soil
            
            # Calculate active pressure at the base of the layer
            pressure = Ka * gamma * height + (Ka * self.surcharge if y_offset == 0 else 0)
            active_pressures.append(active_pressures[-1] + pressure)
            y_offset += height
            depths.append(y_offset)
        
        # Passive Pressure Calculation from bottom to top, starting from the bottom layer depth
        passive_depths = [self.total_depth, self.total_depth - self.passive_layer['Depth']]
        passive_pressures = [0]
        passive_K = self.calculate_earth_pressure_coefficient(self.passive_layer['Friction Angle'], is_passive=True)
        
        # Calculate passive pressure triangle only starting from bottom up
        for depth in passive_depths[:-1]:
            pressure = passive_K * self.passive_layer['Unit Weight'] * (self.total_depth - depth)
            passive_pressures.insert(0, pressure)
        
        # Interpolate active and passive pressures to create smooth curves
        max_depth = max(self.total_depth, self.total_depth - self.passive_layer['Depth'])
        interp_depths = np.linspace(0, max_depth, num=100)
        active_interp = interp1d(depths, active_pressures, kind='linear', fill_value="extrapolate")(interp_depths)
        passive_interp = interp1d(passive_depths, passive_pressures, kind='linear', fill_value="extrapolate")(interp_depths)
        
        # Plot active and passive pressures on the same diagram
        ax.plot(active_interp, interp_depths, label="Active Pressure", color="red")
        ax.plot(passive_interp, interp_depths, label="Passive Pressure", color="blue")

        # Surcharge line at top
        ax.plot([0, active_interp[0]], [0, 0], 'k-', lw=2, label='Surcharge Load')
        
        # Groundwater level line
        if self.groundwater_level < self.total_depth:
            ax.plot([-1.5 * max(passive_interp), 1.5 * max(active_interp)], [self.groundwater_level, self.groundwater_level], 'b--', label='Groundwater Level')

        ax.set_xlabel("Pressure (kPa)")
        ax.set_ylabel("Depth (m)")
        ax.set_title("Combined Active and Passive Earth Pressure Diagram")
        ax.legend()
        ax.grid()
        st.pyplot(fig)  # Display plot in Streamlit

# Streamlit app setup
st.title("Sheet Pile Analysis")

# Material selection
st.subheader("Material Properties")
material_type = st.selectbox("Select Material Type", options=list(material_properties.keys()))
material = material_properties[material_type]
st.write("Material Properties:")
for prop, value in material.items():
    st.write(f"{prop}: {value}")

# Inputs for soil properties
st.subheader("Active Soil Layers")
soil_layers = []
num_layers = st.number_input("Enter number of active soil layers", min_value=1, max_value=5, value=2)

for i in range(num_layers):
    st.write(f"Properties for Active Soil Layer {i + 1}")
    soil_type = st.selectbox(f"Select Soil Type for Layer {i + 1}", options=list(soil_properties.keys()), key=f"soil_type_{i}")
    soil = soil_properties[soil_type]
    
    # Display and allow adjustment of default properties
    unit_weight = st.number_input(f"  Unit Weight of Layer {i + 1} (kN/m³): ", value=soil["Unit Weight"], min_value=1.0)
    friction_angle = st.number_input(f"  Friction Angle of Layer {i + 1} (°): ", value=soil["Friction Angle"], min_value=0.0, max_value=45.0)
    cohesion = st.number_input(f"  Cohesion of Layer {i + 1} (kPa): ", value=soil["Cohesion"], min_value=0.0)
    depth = st.number_input(f"  Depth of Layer {i + 1} (m): ", min_value=1.0)
    soil_layers.append({"Unit Weight": unit_weight, "Friction Angle": friction_angle, "Cohesion": cohesion, "Depth": depth})

# Inputs for passive soil layer
st.subheader("Passive Soil Layer Properties")
passive_soil_type = st.selectbox("Select Passive Soil Type", options=list(soil_properties.keys()), key="passive_soil_type")
passive_soil = soil_properties[passive_soil_type]

# Display and allow adjustment of default properties
passive_unit_weight = st.number_input("Unit Weight of Passive Soil (kN/m³): ", value=passive_soil["Unit Weight"], min_value=1.0)
passive_friction_angle = st.number_input("Friction Angle of Passive Soil (°): ", value=passive_soil["Friction Angle"], min_value=0.0, max_value=45.0)
passive_cohesion = st.number_input("Cohesion of Passive Soil (kPa): ", value=passive_soil["Cohesion"], min_value=0.0)
passive_depth = st.number_input("Depth of Passive Soil (m): ", min_value=1.0)
passive_layer = {"Unit Weight": passive_unit_weight, "Friction Angle": passive_friction_angle, "Cohesion": passive_cohesion, "Depth": passive_depth}

# Additional parameters
surcharge = st.number_input("Surcharge Load (kN/m²): ", min_value=0.0)
groundwater_level = st.number_input("Groundwater Level Depth from Top (m): ", min_value=0.0)

# Initialize analysis and plot the pressure diagram
analysis = SheetPileAnalysis(material, soil_layers, passive_layer, surcharge, groundwater_level)
st.subheader("Pressure Diagram")
analysis.plot_pressure_diagram()
