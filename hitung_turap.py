import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

# Default properties for materials and soils
material_properties = {
    "steel": {"Density": 7850.0, "Yield Strength": 250.0, "Modulus of Elasticity": 200000.0},
    "concrete": {"Density": 2400.0, "Compressive Strength": 30.0, "Modulus of Elasticity": 30000.0}
}

soil_properties = {
    "sand": {"Unit Weight": 18.0, "Friction Angle": 35.0, "Cohesion": 0.0},
    "clay": {"Unit Weight": 20.0, "Friction Angle": 0.0, "Cohesion": 25.0},
    "sandy clay": {"Unit Weight": 19.0, "Friction Angle": 20.0, "Cohesion": 10.0}  # Example properties for sandy clay
}

class SheetPileAnalysis:
    def __init__(self, material, soil_layers, passive_layer, surcharge, groundwater_level, method, safety_factor_threshold=1.5):
        self.material = material
        self.soil_layers = soil_layers
        self.passive_layer = passive_layer
        self.surcharge = surcharge  # Uniform surcharge load (kN/m²)
        self.groundwater_level = groundwater_level  # Depth of groundwater level from top (m)
        self.method = method.lower()
        self.safety_factor_threshold = safety_factor_threshold
        self.total_depth = sum(layer['Depth'] for layer in soil_layers)  # Total depth of active soil

    def calculate_earth_pressure_coefficient(self, friction_angle, is_passive=False):
        if self.method == "rankine":
            angle_factor = (45 + friction_angle / 2) if is_passive else (45 - friction_angle / 2)
            K = np.tan(np.radians(angle_factor)) ** 2
        elif self.method == "coulomb":
            Ka = (np.cos(np.radians(friction_angle)) ** 2) / (
                np.cos(np.radians(friction_angle)) + np.sqrt(np.cos(np.radians(friction_angle - 45))))
            K = Ka if not is_passive else 1 / Ka
        else:
            raise ValueError("Unknown method. Please choose 'Rankine' or 'Coulomb'.")
        return K

    def calculate_stability(self):
        # Sum active and passive forces/moments
        active_moment = 0
        passive_moment = 0
        active_force = 0
        passive_force = 0

        # Calculate active forces and moments
        for layer in self.soil_layers:
            Ka = self.calculate_earth_pressure_coefficient(layer['Friction Angle'])
            gamma = layer['Unit Weight']
            height = layer['Depth']
            if self.groundwater_level < height:
                gamma -= 9.81
            
            pressure = Ka * gamma * height + (Ka * self.surcharge if layer['Depth'] == self.soil_layers[0]['Depth'] else 0)
            force = pressure * height
            moment = force * height / 3  # Moment arm for triangular pressure distribution
            active_force += force
            active_moment += moment
        
        # Calculate passive forces and moments
        passive_K = self.calculate_earth_pressure_coefficient(self.passive_layer['Friction Angle'], is_passive=True)
        passive_force = passive_K * self.passive_layer['Unit Weight'] * self.passive_layer['Depth']
        passive_moment = passive_force * self.passive_layer['Depth'] / 3
        
        # Calculate safety factor
        safety_factor = passive_moment / active_moment
        return active_force, active_moment, passive_force, passive_moment, safety_factor

    def plot_pressure_diagram(self, safety_factor):
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

        # Surcharge arrow at top
        ax.arrow(0, -1, 0, -0.5, head_width=5, head_length=0.5, fc='purple', ec='purple')
        ax.text(0, -2, "Surcharge Load", ha='center', va='center', color='purple')

        # Groundwater level line
        if self.groundwater_level < self.total_depth:
            ax.plot([-1.5 * max(passive_interp), 1.5 * max(active_interp)], [self.groundwater_level, self.groundwater_level], 'b--', label='Groundwater Level')

        # Safety Indicator
        if safety_factor >= self.safety_factor_threshold:
            ax.text(0.5, 0.5, "Safe", color="green", transform=ax.transAxes, fontsize=20, fontweight='bold', ha='center', va='center')
        else:
            ax.text(0.5, 0.5, "Not Safe", color="red", transform=ax.transAxes, fontsize=20, fontweight='bold', ha='center', va='center')

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

# Method selection
method = st.selectbox("Select Analysis Method", options=["Rankine", "Coulomb"])

# Inputs for soil properties
st.subheader("Active Soil Layers")
soil_layers = []
num_layers = st.number_input("Enter number of active soil layers", min_value=1, max_value=5, value=2)

for i in range(num_layers):
    st.write(f"Properties for Active Soil Layer {i + 1}")
    soil_type = st.selectbox(f"Select Soil Type for Layer {i + 1}", options=list(soil_properties.keys()), key=f"soil_type_{i}")
    soil = soil_properties[soil_type]
    
    # Display and allow adjustment of default properties
    unit_weight = st.number_input(f"  Unit Weight of Layer {i + 1} (kN/m³): ", value=float(soil["Unit Weight"]), min_value=1.0)
    friction_angle = st.number_input(f"  Friction Angle of Layer {i + 1} (°): ", value=float(soil["Friction Angle"]), min_value=0.0, max_value=45.
