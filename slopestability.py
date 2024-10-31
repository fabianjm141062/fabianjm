import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate FS using Bishop's simplified method
def calculate_fs_bishop(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices, tolerance=0.001, max_iterations=100):
    FS = 1.0  # Initial guess for FS
    slice_width = slope_height / num_slices
    R = slope_height / np.sin(slope_angle)  # Radius of the circular slip surface

    for _ in range(max_iterations):
        numerator_sum = 0
        denominator_sum = 0
        for i in range(num_slices):
            x = (i + 0.5) * slice_width
            theta = np.arctan(slice_width / R)
            height_slice = slope_height - x * np.tan(slope_angle)
            weight = unit_weight * slice_width * height_slice
            normal_force = weight * np.cos(theta) / (FS + (np.tan(friction_angle) * np.sin(theta) / FS))
            shear_resistance = cohesion * slice_width + normal_force * np.tan(friction_angle)
            numerator_sum += shear_resistance
            denominator_sum += weight * np.sin(theta)
        
        new_FS = numerator_sum / denominator_sum
        if abs(new_FS - FS) < tolerance:
            return new_FS  # Converged FS
        FS = new_FS

    return FS  # Return FS if maximum iterations reached

# Function to plot slope and failure surface
def plot_slope(slope_height, slope_angle, R, FS, num_slices):
    slope_width = slope_height / np.tan(slope_angle)
    x_slope = [0, slope_width]
    y_slope = [0, slope_height]

    # Adjust failure surface to end within the slope profile
    center_x = slope_width / 2
    center_y = 0  # Set center to start at slope base
    theta = np.linspace(0, np.pi, 100)
    x_circle = center_x + R * np.cos(theta)
    y_circle = center_y + R * np.sin(theta)

    plt.figure(figsize=(10, 6))
    plt.plot(x_slope, y_slope, color='black', linewidth=2, label='Slope Surface')
    plt.plot(x_circle, y_circle, color='red', linestyle='--', label='Failure Surface')

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
st.title("Slope Stability Analysis Using Bishop Method by Fabian J Manoppo")

# Input parameters
slope_height = st.number_input("Slope Height (m)", min_value=1.0, value=10.0)
slope_angle = np.radians(st.number_input("Slope Angle (degrees)", min_value=1.0, max_value=90.0, value=30.0))
cohesion = st.number_input("Cohesion (ton/m²)", min_value=0.0, value=3.2)
unit_weight = st.number_input("Unit Weight (ton/m³)", min_value=0.0, value=1.8)
friction_angle = np.radians(st.number_input("Friction Angle (degrees)", min_value=0.0, max_value=45.0, value=20.0))
num_slices = st.number_input("Number of Slices", min_value=1, max_value=50, value=10)

# Calculate FS
R = slope_height / np.sin(slope_angle)  # Radius of the circular slip surface
fs_bishop = calculate_fs_bishop(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices)
st.write(f"Calculated Factor of Safety (FS) using Bishop Method: {fs_bishop:.3f}")

# Plot slope and failure surface
plot_slope(slope_height, slope_angle, R, fs_bishop, num_slices)
