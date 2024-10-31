import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Material properties for selection
material_properties = {
    "steel": {"Density": 7850.0, "Yield Strength": 250.0, "Modulus of Elasticity": 200000.0},
    "concrete": {"Density": 2400.0, "Compressive Strength": 30.0, "Modulus of Elasticity": 30000.0}
}

# Single soil type: "sandy clay"
soil_properties = {
    "sandy clay": {"Unit Weight": 19.0, "Friction Angle": 20.0, "Cohesion": 10.0}
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
            phi = np.radians(friction_angle)
            if is_passive:
                K = (np.cos(phi) + np.sqrt(np.cos(phi)**2 - np.cos(np.radians(45 - friction_angle / 2))**2)) / np.cos(phi)
            else:
                K = (np.cos(phi) - np.sqrt(np.cos(phi)**2 - np.cos(np.radians(45 + friction_angle / 2))**2)) / np.cos(phi)
        else:
            raise ValueError("Unknown method. Please choose 'Rankine' or 'Coulomb'.")
        return K

    def plot_pressure_diagram(self):
        fig, ax = plt.subplots(figsize=(8, 10))
        
        # Active Pressure Calculation with surcharge and groundwater effect
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
        
        # Passive Pressure Calculation only from bottom for passive layer depth
        bottom_depth = self.total_depth
        passive_depths = [bottom_depth, bottom_depth - self.passive_layer['Depth']]
        passive_pressures = [0]
        passive_K = self.calculate_earth_pressure_coefficient(self.passive_layer['Friction Angle'], is_passive=True)
        
        for depth in passive_depths[:-1]:
            pressure = passive_K * self.passive_layer['Unit Weight'] * (bottom_depth - depth)
            passive_pressures.insert(0, pressure)
        
        interp_depths = np.linspace(0, self.total_depth, num=100)
        
        # Plot Active and Passive Pressures as Shaded Areas
        ax.fill_betweenx(interp_depths, 0, np.interp(interp_depths, depths, active_pressures), color="red", alpha=0.3, label="Active Pressure")
        ax.fill_betweenx(interp_depths, np.interp(interp_depths, passive_depths, passive_pressures), 0, color="blue", alpha=0.3, label="Passive Pressure")

        # Surcharge Load as a Vertical Line
        ax.axvline(x=self.surcharge, color="black", linewidth=4, label="Surcharge Load")

        # Groundwater Level as a Dashed Line
        ax.axhline(y=self.groundwater_level, color="blue", linestyle="--", label="Groundwater Level")
        ax.text(self.surcharge, self.groundwater_level, "Muka Air Tanah", color="blue", va="bottom", ha="right")

        ax.set_xlabel("Pressure (kPa)")
        ax.set_ylabel("Depth (m)")
        ax.invert_yaxis()  # Ensures the depth axis starts from 0 at the top and increases downward
        ax.set_title("Diagram Tekanan Tanah Aktif dan Pasif dengan Surcharge dan Muka Air Tanah")
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

# All soil layers are of type "sandy clay"
soil = soil_properties["sandy clay"]
soil_layers = []
num_layers = st.number_input("Enter number of active soil layers", min_value=1, max_value=5, value=2)

for i in range(num_layers):
    st.write(f"Properties for Active Soil Layer {i + 1}")
    unit_weight = st.number_input(f"  Unit Weight of Layer {i + 1} (kN/m³): ", value=float(soil["Unit Weight"]), min_value=1.0)
    friction_angle = st.number_input(f"  Friction Angle of Layer {i + 1} (°): ", value=float(soil["Friction Angle"]), min_value=0.0, max_value=45.0)
    cohesion = st.number_input(f"  Cohesion of Layer {i + 1} (kPa): ", value=float(soil["Cohesion"]), min_value=0.0)
    depth = st.number_input(f"  Depth of Layer {i + 1} (m): ", min_value=1.0)
    soil_layers.append({"Unit Weight": unit_weight, "Friction Angle": friction_angle, "Cohesion": cohesion, "Depth": depth})

# Passive soil layer also of type "sandy clay"
st.subheader("Passive Soil Layer Properties")
passive_unit_weight = st.number_input("Unit Weight of Passive Soil (kN/m³): ", value=float(soil["Unit Weight"]), min_value=1.0)
passive_friction_angle = st.number_input("Friction Angle of Passive Soil (°): ", value=float(soil["Friction Angle"]), min_value=0.0, max_value=45.0)
passive_cohesion = st.number_input("Cohesion of Passive Soil (kPa): ", value=float(soil["Cohesion"]), min_value=0.0)
passive_depth = st.number_input("Depth of Passive Soil (m): ", min_value=1.0)
passive_layer = {"Unit Weight": passive_unit_weight, "Friction Angle": passive_friction_angle, "Cohesion": passive_cohesion, "Depth": passive_depth}

st.subheader("Additional Parameters")
surcharge = st.number_input("Surcharge Load (kN/m²): ", min_value=0.0)
groundwater_level = st.number_input("Groundwater Level Depth from Top (m): ", min_value=0.0)
analysis_method = st.selectbox("Select Analysis Method", options=["Rankine", "Coulomb"])

# Calculate and display results when the button is clicked
if st.button("Calculate"):
    analysis = SheetPileAnalysis(material, soil_layers, passive_layer, surcharge, groundwater_level, analysis_method)
    st.subheader("Pressure Diagram")
    analysis.plot_pressure_diagram()
