import streamlit as st
import pandas as pd
import numpy as np

class SheetPileStability:
    def __init__(self, soil_layers, passive_layer, surcharge_load, groundwater_level, required_safety_factor=1.5):
        self.soil_layers = soil_layers
        self.passive_layer = passive_layer
        self.surcharge_load = surcharge_load
        self.groundwater_level = groundwater_level
        self.required_safety_factor = required_safety_factor
        self.total_depth = sum(layer['Depth'] for layer in soil_layers)

    def calculate_earth_pressure_coefficient(self, friction_angle, is_passive=False):
        if is_passive:
            K = np.tan(np.radians(45 + friction_angle / 2)) ** 2
        else:
            K = np.tan(np.radians(45 - friction_angle / 2)) ** 2
        return K

    def calculate_forces_and_moments(self):
        active_moment = 0
        passive_moment = 0
        y_offset = 0

        for layer in self.soil_layers:
            Ka = self.calculate_earth_pressure_coefficient(layer['Friction Angle'])
            gamma = layer['Unit Weight']
            height = layer['Depth']
            if y_offset + height > self.groundwater_level:
                gamma -= 9.81

            force = 0.5 * Ka * gamma * height ** 2
            moment = force * (y_offset + height / 3)
            active_moment += moment
            y_offset += height

        passive_K = self.calculate_earth_pressure_coefficient(self.passive_layer['Friction Angle'], is_passive=True)
        passive_force = 0.5 * passive_K * self.passive_layer['Unit Weight'] * self.passive_layer['Depth'] ** 2
        passive_moment = passive_force * (self.total_depth - self.passive_layer['Depth'] / 3)

        return active_moment, passive_moment

    def stability_analysis(self):
        active_moment, passive_moment = self.calculate_forces_and_moments()
        safety_factor = abs(passive_moment / active_moment) if active_moment != 0 else 0
        stability = "Safe" if safety_factor >= self.required_safety_factor else "Unsafe"

        results_df = pd.DataFrame({
            'Total Active Moment (kNm)': [active_moment],
            'Total Passive Moment (kNm)': [passive_moment],
            'Safety Factor': [round(safety_factor, 2)],
            'Stability': [stability]
        })

        return results_df

# Streamlit UI for input
st.title("Sheet Pile Stability Control by Fabian J Manoppo")

# Input for soil properties (active layers)
soil_layers = []
num_layers = st.number_input("Enter number of active soil layers", min_value=1, max_value=5, value=2)

for i in range(num_layers):
    st.write(f"Properties for Active Soil Layer {i + 1}")
    unit_weight = st.number_input(f"Unit Weight of Layer {i + 1} (kN/m³): ", min_value=1.0)
    friction_angle = st.number_input(f"Friction Angle of Layer {i + 1} (°): ", min_value=0.0, max_value=45.0)
    depth = st.number_input(f"Depth of Layer {i + 1} (m): ", min_value=1.0)
    soil_layers.append({"Unit Weight": unit_weight, "Friction Angle": friction_angle, "Depth": depth})

# Input for passive soil properties
st.subheader("Passive Soil Layer Properties")
passive_unit_weight = st.number_input("Unit Weight of Passive Soil (kN/m³): ", min_value=1.0)
passive_friction_angle = st.number_input("Friction Angle of Passive Soil (°): ", min_value=0.0, max_value=45.0)
passive_depth = st.number_input("Depth of Passive Soil (m): ", min_value=1.0)
passive_layer = {"Unit Weight": passive_unit_weight, "Friction Angle": passive_friction_angle, "Depth": passive_depth}

# Additional parameters
surcharge_load = st.number_input("Surcharge Load (kN/m²): ", min_value=0.0)
groundwater_level = st.number_input("Groundwater Level Depth from Top (m): ", min_value=0.0)

# Initialize and calculate stability
design = SheetPileStability(soil_layers, passive_layer, surcharge_load, groundwater_level)

if st.button("Check Stability"):
    stability_results = design.stability_analysis()
    st.subheader("Stability Analysis Results")
    st.dataframe(stability_results)
