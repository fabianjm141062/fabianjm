import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Material properties for sheet pile selection
material_properties = {
    "Steel": {"Density": 7850.0, "Yield Strength": 250.0, "Modulus of Elasticity": 200000.0},
    "Prestressed Concrete": {"Density": 2400.0, "Compressive Strength": 30.0, "Modulus of Elasticity": 30000.0}
}

# Different soil types and their properties
soil_types = {
    "Sand": {"Unit Weight": 18.0, "Friction Angle": 30.0, "Cohesion": 0.0},
    "Clay": {"Unit Weight": 17.0, "Friction Angle": 0.0, "Cohesion": 25.0},
    "Sandy Clay": {"Unit Weight": 19.0, "Friction Angle": 20.0, "Cohesion": 10.0}
}

class SheetPileDesign:
    def __init__(self, material, soil_layers, passive_layer, surcharge_load, groundwater_level, required_safety_factor=1.5):
        self.material = material
        self.soil_layers = soil_layers
        self.passive_layer = passive_layer
        self.surcharge_load = surcharge_load
        self.groundwater_level = groundwater_level
        self.required_safety_factor = required_safety_factor
        self.total_depth = sum(layer['Depth'] for layer in soil_layers)  # Total depth of active soil

    def calculate_earth_pressure_coefficient(self, friction_angle, is_passive=False):
        # Using Rankine's formula for simplicity
        if is_passive:
            K = np.tan(np.radians(45 + friction_angle / 2)) ** 2
        else:
            K = np.tan(np.radians(45 - friction_angle / 2)) ** 2
        return K

    def calculate_forces_and_moments(self):
        # Calculate active forces and moments
        active_forces = []
        active_moments = []
        y_offset = 0  # Starting depth

        for layer in self.soil_layers:
            Ka = self.calculate_earth_pressure_coefficient(layer['Friction Angle'])
            gamma = layer['Unit Weight']
            height = layer['Depth']

            # Adjust for submerged weight if layer is below groundwater level
            if y_offset + height > self.groundwater_level:
                gamma -= 9.81  # Reduce by the unit weight of water

            # Calculate active force and moment for each layer
            force = 0.5 * Ka * gamma * height ** 2
            moment = force * (y_offset + height / 3)  # Moment about the bottom
            active_forces.append(force)
            active_moments.append(moment)
            y_offset += height

        total_active_force = sum(active_forces)
        total_active_moment = sum(active_moments)

        # Calculate passive forces and moments
        passive_K = self.calculate_earth_pressure_coefficient(self.passive_layer['Friction Angle'], is_passive=True)
        passive_force = 0.5 * passive_K * self.passive_layer['Unit Weight'] * self.passive_layer['Depth'] ** 2
        passive_moment = passive_force * (self.total_depth - self.passive_layer['Depth'] / 3)

        return total_active_force, total_active_moment, passive_force, passive_moment

    def stability_analysis(self):
        # Calculate forces and moments
        total_active_force, total_active_moment, passive_force, passive_moment = self.calculate_forces_and_moments()

        # Calculate the safety factor
        safety_factor = passive_moment / total_active_moment
        stability = "Safe" if safety_factor >= self.required_safety_factor else "Unsafe"

        # Summary of results
        results = {
            'Total Active Force (kN)': total_active_force,
            'Total Passive Force (kN)': passive_force,
            'Total Active Moment (kNm)': total_active_moment,
            'Total Passive Moment (kNm)': passive_moment,
            'Safety Factor': safety_factor,
            'Stability': stability
        }
        return pd.DataFrame([results])

    def plot_pressure_diagram(self):
        fig, ax = plt.subplots(figsize=(6, 8))
        ax.set_xlim(-1.5 * self.passive_layer['Depth'], 1.5 * self.total_depth)  # Set x-axis limits
        ax.set_ylim(0, self.total_depth)  # Set y-axis limits for depth

        # Draw the sheet pile as a black rectangle
        ax.plot([0, 0], [0, self.total_depth], color='black', linewidth=8)

        # Plot active and passive pressures as triangles
        active_base = self.calculate_earth_pressure_coefficient(self.soil_layers[-1]['Friction Angle']) * self.soil_layers[-1]['Unit Weight'] * self.total_depth
        passive_base = self.calculate_earth_pressure_coefficient(self.passive_layer['Friction Angle'], is_passive=True) * self.passive_layer['Unit Weight'] * self.passive_layer['Depth']
        ax.fill_betweenx([0, self.total_depth], 0, active_base, color='red', alpha=0.3, label='Active Pressure')
        ax.fill_betweenx([self.total_depth - self.passive_layer['Depth'], self.total_depth], 0, -passive_base, color='blue', alpha=0.3, label='Passive Pressure')

        # Surcharge load as a horizontal line on the active side
        ax.plot([0, active_base], [0, 0], 'k-', lw=2, label='Surcharge Load (q)')

        # Groundwater level as a dashed blue line
        if self.groundwater_level < self.total_depth:
            ax.plot([-1.5 * passive_base, 1.5 * active_base], [self.groundwater_level, self.groundwater_level], 'b--', label='Groundwater Level')

        # Set labels and title
        ax.set_xlabel("Pressure (kPa)")
        ax.set_ylabel("Depth (m)")
        ax.set_title("Active and Passive Earth Pressure Diagram with Surcharge and Groundwater Level")

        plt.gca().invert_yaxis()  # Invert y-axis to represent depth
        plt.grid()
        plt.legend()
        st.pyplot(fig)

# Streamlit UI for input
st.title("Sheet Pile Design Stability Analysis")

# Material selection for the sheet pile
st.subheader("Select Sheet Pile Material")
material_type = st.selectbox("Material Type", options=list(material_properties.keys()))
material = material_properties[material_type]
st.write("Material Properties:")
for prop, value in material.items():
    st.write(f"{prop}: {value}")

# Input for soil properties
soil_layers = []
num_layers = st.number_input("Enter number of active soil layers", min_value=1, max_value=5, value=2)

for i in range(num_layers):
    st.write(f"Properties for Active Soil Layer {i + 1}")
    soil_type = st.selectbox(f"Select Soil Type for Layer {i + 1}", options=list(soil_types.keys()), key=f"soil_type_{i}")
    unit_weight = soil_types[soil_type]["Unit Weight"]
    friction_angle = soil_types[soil_type]["Friction Angle"]
    cohesion = soil_types[soil_type]["Cohesion"]
    depth = st.number_input(f"Depth of Layer {i + 1} (m): ", min_value=1.0)
    soil_layers.append({"Unit Weight": unit_weight, "Friction Angle": friction_angle, "Cohesion": cohesion, "Depth": depth})

# Passive soil layer properties with soil type selection
st.subheader("Passive Soil Layer Properties")
passive_soil_type = st.selectbox("Select Passive Soil Type", options=list(soil_types.keys()))
passive_unit_weight = soil_types[passive_soil_type]["Unit Weight"]
passive_friction_angle = soil_types[passive_soil_type]["Friction Angle"]
passive_cohesion = soil_types[passive_soil_type]["Cohesion"]
passive_depth = st.number_input("Depth of Passive Soil (m): ", min_value=1.0)
passive_layer = {"Unit Weight": passive_unit_weight, "Friction Angle": passive_friction_angle, "Cohesion": passive_cohesion, "Depth": passive_depth}

# Additional parameters
surcharge_load = st.number_input("Surcharge Load (kN/m²): ", min_value=0.0)
groundwater_level = st.number_input("Groundwater Level Depth from Top (m): ", min_value=0.0)

# Initialize and calculate stability
design = SheetPileDesign(material, soil_layers, passive_layer, surcharge_load, groundwater_level)

if st.button("Calculate"):
    stability_results = design.stability_analysis()
    st.subheader("Stability Analysis Results")
    st.dataframe(stability_results)

    # Plot the pressure diagram
    st.subheader("Pressure Diagram")
    design.plot_pressure_diagram()
