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
    def __init__(self, material, soil_layers, passive_layer, surcharge, groundwater_level, analysis_method, safety_factor_threshold=1.5):
        self.material = material
        self.soil_layers = soil_layers
        self.passive_layer = passive_layer
        self.surcharge = surcharge
        self.groundwater_level = groundwater_level
        self.analysis_method = analysis_method.lower()
        self.safety_factor_threshold = safety_factor_threshold
        self.total_depth = sum(layer['Depth'] for layer in soil_layers)

    def calculate_earth_pressure_coefficient(self, friction_angle, is_passive=False):
        if self.analysis_method == "rankine":
            angle_factor = (45 + friction_angle / 2) if is_passive else (45 - friction_angle / 2)
            K = np.tan(np.radians(angle_factor)) ** 2
        elif self.analysis_method == "coulomb":
            if is_passive:
                K = (np.cos(np.radians(friction_angle)) ** 2) / (
                    np.cos(np.radians(friction_angle)) + np.sqrt(np.cos(np.radians(friction_angle - 45))))
            else:
                K = (np.cos(np.radians(friction_angle)) ** 2) / (
                    np.cos(np.radians(friction_angle)) - np.sqrt(np.cos(np.radians(friction_angle - 45)))
                )
        else:
            raise ValueError("Unknown method. Please choose 'Rankine' or 'Coulomb'.")
        return K

    def calculate_stability(self):
        active_moment = 0
        passive_moment = 0
        active_force = 0
        passive_force = 0

        for layer in self.soil_layers:
            Ka = self.calculate_earth_pressure_coefficient(layer['Friction Angle'])
            gamma = layer['Unit Weight']
            height = layer['Depth']
            if self.groundwater_level < height:
                gamma -= 9.81
            
            pressure = Ka * gamma * height + (Ka * self.surcharge if layer['Depth'] == self.soil_layers[0]['Depth'] else 0)
            force = pressure * height
            moment = force * height / 3
            active_force += force
            active_moment += moment
        
        passive_K = self.calculate_earth_pressure_coefficient(self.passive_layer['Friction Angle'], is_passive=True)
        passive_force = passive_K * self.passive_layer['Unit Weight'] * self.passive_layer['Depth']
        passive_moment = passive_force * self.passive_layer['Depth'] / 3
        
        safety_factor = passive_moment / active_moment
        return active_force, active_moment, passive_force, passive_moment, safety_factor

    def plot_pressure_diagram(self, safety_factor):
        fig, ax = plt.subplots(figsize=(8, 10))
        ax.invert_yaxis()
        
        depths = [0]
        active_pressures = [0]
        y_offset = 0
        
        for layer in self.soil_layers:
            Ka = self.calculate_earth_pressure_coefficient(layer['Friction Angle'])
            gamma = layer['Unit Weight']
            height = layer['Depth']
            if y_offset + height > self.groundwater_level:
                gamma -= 9.81
            
            pressure = Ka * gamma * height + (Ka * self.surcharge if y_offset == 0 else 0)
            active_pressures.append(active_pressures[-1] + pressure)
            y_offset += height
            depths.append(y_offset)
        
        passive_depths = [self.total_depth, self.total_depth - self.passive_layer['Depth']]
        passive_pressures = [0]
        passive_K = self.calculate_earth_pressure_coefficient(self.passive_layer['Friction Angle'], is_passive=True)
        
        for depth in passive_depths[:-1]:
            pressure = passive_K * self.passive_layer['Unit Weight'] * (self.total_depth - depth)
            passive_pressures.insert(0, pressure)
        
        max_depth = max(self.total_depth, self.total_depth - self.passive_layer['Depth'])
        interp_depths = np.linspace(0, max_depth, num=100)
        active_interp = interp1d(depths, active_pressures, kind='linear', fill_value="extrapolate")(interp_depths)
        passive_interp = interp1d(passive_depths, passive_pressures, kind='linear', fill_value="extrapolate")(interp_depths)
        
        ax.plot(active_interp, interp_depths, label="Active Pressure", color="red")
        ax.plot(passive_interp, interp_depths, label="Passive Pressure", color="blue")

        ax.plot([0, 0], [0, -self.surcharge], 'g|-', linewidth=2, label='Surcharge Load')  # Surcharge arrow line
        
        if self.groundwater_level < self.total_depth:
            ax.plot([-1.5 * max(passive_interp), 1.5 * max(active_interp)], [self.groundwater_level, self.groundwater_level], 'b--', label='Groundwater Level')

        if safety_factor >= self.safety_factor_threshold:
            ax.text(0.5, 0.5, "Safe", color="green", transform=ax.transAxes, fontsize=20, fontweight='bold', ha='center', va='center')
        else:
            ax.text(0.5, 0.5, "Not Safe", color="red", transform=ax.transAxes, fontsize=20, fontweight='bold', ha='center', va='center')

        ax.set_xlabel("Pressure (kPa)")
        ax.set_ylabel("Depth (m)")
        ax.set_title("Combined Active and Passive Earth Pressure Diagram")
        ax.legend()
        ax.grid()
        st.pyplot(fig)

st.title("Sheet Pile Analysis")

st.subheader("Material Properties")
material_type = st.selectbox("Select Material Type", options=list(material_properties.keys()))
material = material_properties[material_type]
st.write("Material Properties:")
for prop, value in material.items():
    st.write(f"{prop}: {value}")

st.subheader("Active Soil Layers")
soil_layers = []
num_layers = st.number_input("Enter number of active soil layers", min_value=1, max_value=5, value=2)

for i in range(num_layers):
    st.write(f"Properties for Active Soil Layer {i + 1}")
    soil_type = st.selectbox(f"Select Soil Type for Layer {i + 1}", options=list(soil_properties.keys()), key=f"soil_type_{i}")
    soil = soil_properties[soil_type]
    
    unit_weight = st.number_input(f"  Unit Weight of Layer {i + 1} (kN/m³): ", value=float(soil["Unit Weight"]), min_value=1.0)
    friction_angle = st.number_input(f"  Friction Angle of Layer {i + 1} (°): ", value=float(soil["Friction Angle"]), min_value=0.0, max_value=45.0)
    cohesion = st.number_input(f"  Cohesion of Layer {i + 1} (kPa): ", value=float(soil["Cohesion"]), min_value=0.0)
    depth = st.number_input(f"  Depth of Layer {i + 1} (m): ", min_value=1.0)
    soil_layers.append({"Unit Weight": unit_weight, "Friction Angle": friction_angle, "Cohesion": cohesion, "Depth": depth})

st.subheader("Passive Soil Layer Properties")
passive_soil_type = st.selectbox("Select Passive Soil Type", options=list(soil_properties.keys()), key="passive_soil_type")
passive_soil = soil_properties[passive_soil_type]

passive_unit_weight = st.number_input("Unit Weight of Passive Soil (kN/m³): ", value=float(passive_soil["Unit Weight"]), min_value=1.0)
passive_friction_angle
