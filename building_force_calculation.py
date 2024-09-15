import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Title of the app
st.title("Internal Forces and Moments of Multi-Story Building by Fabian J Manoppo")

# Input parameters for the structure
material = st.selectbox("Select Material", ["Steel", "Reinforced Concrete"])
num_columns = st.number_input("Number of Columns", min_value=1, value=4)
num_floors = st.number_input("Number of Floors", min_value=1, value=10)
floor_height = st.number_input("Floor Height (m)", min_value=2.5, value=3.0)
building_width = st.number_input("Building Width (m)", min_value=5.0, value=10.0)
building_length = st.number_input("Building Length (m)", min_value=5.0, value=10.0)

# Input loads (in kN/m²)
dead_load = st.number_input("Dead Load (kN/m²)", value=5.0)
live_load = st.number_input("Live Load (kN/m²)", value=2.5)
earthquake_load = st.number_input("Earthquake Load (kN/m²)", value=1.0)
wind_load = st.number_input("Wind Load (kN/m²)", value=0.5)

# Material properties (simplified)
if material == "Steel":
    E = 210 * 10**3  # Modulus of elasticity for steel (MPa)
    yield_strength = 250  # Yield strength of steel (MPa)
elif material == "Reinforced Concrete":
    E = 25 * 10**3  # Modulus of elasticity for concrete (MPa)
    yield_strength = 30  # Yield strength of concrete (MPa)

# Calculation of forces (simplified)
total_height = num_floors * floor_height
total_dead_load = dead_load * building_width * building_length
total_live_load = live_load * building_width * building_length
total_eq_load = earthquake_load * building_width * building_length
total_wind_load = wind_load * building_width * building_length

# Calculate vertical forces (dead load + live load), normal forces, and moments
vertical_force = total_dead_load + total_live_load
moment_base = (total_eq_load * total_height) / 2  # Simplified moment calculation due to earthquake load
wind_moment = (total_wind_load * total_height) / 2  # Simplified moment calculation due to wind load

# Display results
st.subheader("Results:")
st.write(f"Total Vertical Force (kN): {vertical_force:.2f}")
st.write(f"Moment at Base due to Earthquake Load (kNm): {moment_base:.2f}")
st.write(f"Moment at Base due to Wind Load (kNm): {wind_moment:.2f}")

# Plot 3D building structure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Coordinates for the columns
x_vals = np.linspace(0, building_width, int(np.sqrt(num_columns)))
y_vals = np.linspace(0, building_length, int(np.sqrt(num_columns)))
x_vals, y_vals = np.meshgrid(x_vals, y_vals)
z_vals = np.zeros_like(x_vals)

# Plot the columns as vertical lines
for i in range(x_vals.shape[0]):
    for j in range(x_vals.shape[1]):
        ax.plot([x_vals[i, j], x_vals[i, j]], [y_vals[i, j], y_vals[i, j]], [0, total_height], 'b-')

# Labels and dimensions
ax.set_xlabel("Width (m)")
ax.set_ylabel("Length (m)")
ax.set_zlabel("Height (m)")
ax.set_title("3D Building Structure")

# Show the 3D plot
st.pyplot(fig)
