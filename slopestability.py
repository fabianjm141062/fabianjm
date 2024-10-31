import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Bishop Method Calculation
def calculate_fs_bishop(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices, tolerance=0.001, max_iterations=100):
    FS = 1.0  # Initial guess for FS
    slice_width = slope_height / num_slices
    R = slope_height / np.sin(slope_angle)  # Radius of the circular slip surface
    table_data = []  # To store detailed calculation results

    for _ in range(max_iterations):
        numerator_sum = 0
        denominator_sum = 0
        slice_results = []
        
        # Loop over each slice
        for i in range(num_slices):
            x = (i + 0.5) * slice_width
            theta = np.arctan(slice_width / R)
            height_slice = slope_height - x * np.tan(slope_angle)
            weight = unit_weight * slice_width * height_slice
            
            # Calculate normal and shear forces
            normal_force = weight * np.cos(theta) / (FS + (np.tan(friction_angle) * np.sin(theta) / FS))
            shear_resistance = cohesion * slice_width + normal_force * np.tan(friction_angle)
            
            numerator_sum += shear_resistance
            denominator_sum += weight * np.sin(theta)
            
            slice_results.append({"Slice": i + 1, "Weight (W)": weight, "Normal Force (N)": normal_force, "Shear Resistance (T)": shear_resistance})
        
        # Update FS
        new_FS = numerator_sum / denominator_sum
        if abs(new_FS - FS) < tolerance:
            FS = new_FS
            table_data = slice_results
            break
        FS = new_FS

    return FS, pd.DataFrame(table_data)

# Culmann's Method Calculation (planar failure)
def calculate_fs_culmann(cohesion, unit_weight, friction_angle, slope_height, slope_angle):
    slope_angle_deg = np.degrees(slope_angle)
    critical_angle = 45 - (friction_angle / 2)  # Critical failure angle in degrees
    FS = cohesion / (unit_weight * slope_height * np.sin(np.radians(slope_angle_deg - critical_angle)) * np.cos(np.radians(critical_angle)))
    return FS

# Taylor's Method Calculation (circular failure)
def calculate_fs_taylor(cohesion, unit_weight, friction_angle, slope_height):
    N = 0.261  # Taylor's stability number for φ = 20 degrees and homogeneous slope
    FS = cohesion / (unit_weight * slope_height * N)
    return FS

# Hoek-Brown Criterion for Rock Slopes
def calculate_fs_hoek_brown(rock_strength, mi, disturbance_factor, unit_weight, slope_height):
    GSI = 60  # Geologic Strength Index (example value)
    mb = mi * np.exp((GSI - 100) / 28)
    s = np.exp((GSI - 100) / 9)
    a = 0.5
    FS = rock_strength / (unit_weight * slope_height * (mb * disturbance_factor * s) ** a)
    return FS

# Function to plot slope and failure surface (for Bishop)
def plot_slope(slope_height, slope_angle, R, FS, num_slices):
    slope_width = slope_height / np.tan(slope_angle)
    x_slope = [0, slope_width]
    y_slope = [0, slope_height]

    center_x = slope_width / 2
    center_y = 0  # Set center to start at slope base
    theta = np.linspace(0, np.pi, 100)
    x_circle = center_x + R * np.cos(theta)
    y_circle = center_y + R * np.sin(theta)

    plt.figure(figsize=(10, 6))
    plt.plot(x_slope, y_slope, color='black', linewidth=2, label='Slope Surface')
    plt.plot(x_circle, y_circle, color='red', linestyle='--', label='Bishop Failure Surface')

    slice_width = slope_width / num_slices
    for i in range(num_slices):
        x = i * slice_width
        y = x * np.tan(slope_angle)
        plt.plot([x, x], [0, y], color='blue', linestyle=':', linewidth=1)

    plt.title(f"Slope Stability Analysis (Bishop Method) - FS: {FS:.3f}")
    plt.xlabel("Width (m)")
    plt.ylabel("Height (m)")
    plt.legend()
    plt.grid()
    st.pyplot(plt)

# Streamlit application
st.title("Slope Stability Analysis Using Multiple Methods by Fabian J Manoppo")

# Input parameters
slope_height = st.number_input("Slope Height (m)", min_value=1.0, value=10.0)
slope_angle = np.radians(st.number_input("Slope Angle (degrees)", min_value=1.0, max_value=90.0, value=30.0))
cohesion = st.number_input("Cohesion (ton/m²)", min_value=0.0, value=3.2)
unit_weight = st.number_input("Unit Weight (ton/m³)", min_value=0.0, value=1.8)
friction_angle = np.radians(st.number_input("Friction Angle (degrees)", min_value=0.0, max_value=45.0, value=20.0))
num_slices = st.number_input("Number of Slices for Bishop Method", min_value=1, max_value=50, value=10)
rock_strength = st.number_input("Rock Strength for Hoek-Brown (MPa)", min_value=0.0, value=10.0)
mi = st.number_input("Hoek-Brown Material Constant (mi)", min_value=0.0, value=10.0)
disturbance_factor = st.number_input("Hoek-Brown Disturbance Factor (D)", min_value=0.0, max_value=1.0, value=0.5)

# Calculate FS for each method
R = slope_height / np.sin(slope_angle)  # Radius of the circular slip surface for Bishop
fs_bishop, calculation_table = calculate_fs_bishop(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices)
fs_culmann = calculate_fs_culmann(cohesion, unit_weight, friction_angle, slope_height, slope_angle)
fs_taylor = calculate_fs_taylor(cohesion, unit_weight, friction_angle, slope_height)
fs_hoek_brown = calculate_fs_hoek_brown(rock_strength, mi, disturbance_factor, unit_weight, slope_height)

# Display FS results
st.write("### Factor of Safety (FS) Calculations")
st.write(f"Bishop Method: {fs_bishop:.3f}")
st.write(f"Culmann Method: {fs_culmann:.3f}")
st.write(f"Taylor Method: {fs_taylor:.3f}")
st.write(f"Hoek-Brown Criterion: {fs_hoek_brown:.3f}")

# Display calculation table for Bishop Method
st.write("### Detailed Calculation Table for Bishop Method (Slice by Slice)")
st.dataframe(calculation_table)

# Plot slope and failure surface for Bishop Method
plot_slope(slope_height, slope_angle, R, fs_bishop, num_slices)
