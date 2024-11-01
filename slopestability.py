import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Bishop Method Calculation with Detailed Steps
def calculate_fs_bishop(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices, tolerance=0.001, max_iterations=100):
    FS = 1.0  # Initial guess for FS
    slice_width = slope_height / num_slices
    R = slope_height
    table_data = []
    calculation_steps = []

    for iteration in range(max_iterations):
        numerator_sum = 0
        denominator_sum = 0
        slice_results = []
        
        for i in range(num_slices):
            x = (i + 0.5) * slice_width
            theta = np.arctan(slice_width / R)
            height_slice = slope_height - x * np.tan(slope_angle)
            weight = unit_weight * slice_width * height_slice
            normal_force = weight * np.cos(theta) / (FS + (np.tan(friction_angle) * np.sin(theta) / FS))
            shear_resistance = cohesion * slice_width + normal_force * np.tan(friction_angle)
            numerator_sum += shear_resistance
            denominator_sum += weight * np.sin(theta)
            slice_results.append({"Slice": i + 1, "Weight (W)": weight, "Normal Force (N)": normal_force, "Shear Resistance (T)": shear_resistance})
        
        new_FS = numerator_sum / denominator_sum
        calculation_steps.append(f"Iteration {iteration + 1}: FS = {new_FS:.4f} (Numerator Sum = {numerator_sum:.4f}, Denominator Sum = {denominator_sum:.4f})")

        if abs(new_FS - FS) < tolerance:
            FS = new_FS
            table_data = slice_results
            break
        FS = new_FS

    return FS, pd.DataFrame(table_data), calculation_steps

# Taylor's Method Calculation
def calculate_fs_taylor(cohesion, unit_weight, friction_angle, slope_height):
    N = 0.261  # Taylor's stability number for φ = 20 degrees and homogeneous slope
    calculation_steps = [
        "Taylor's Stability Number (N) for φ=20 degrees: 0.261",
        f"FS Calculation: FS = {cohesion} / ({unit_weight} * {slope_height} * {N})"
    ]
    FS = cohesion / (unit_weight * slope_height * N)
    calculation_steps.append(f"Resulting FS: {FS:.4f}")
    return FS, calculation_steps

# Culmann's Method Calculation
def calculate_fs_culmann(cohesion, unit_weight, friction_angle, slope_height, slope_angle):
    slope_angle_deg = np.degrees(slope_angle)
    critical_angle = 45 - (np.degrees(friction_angle) / 2)  # Critical failure angle in degrees
    calculation_steps = [
        f"Slope Angle (degrees): {slope_angle_deg:.2f}",
        f"Critical Failure Angle (degrees): {critical_angle:.2f}",
    ]
    FS = cohesion / (unit_weight * slope_height * np.sin(np.radians(slope_angle_deg - critical_angle)) * np.cos(np.radians(critical_angle)))
    calculation_steps.append(f"FS Calculation: {FS:.4f}")
    return FS, calculation_steps

# Hoek-Brown Criterion for Rock Slopes
def calculate_fs_hoek_brown(rock_strength, mi, disturbance_factor, unit_weight, slope_height):
    GSI = 60  # Geologic Strength Index (example value)
    mb = mi * np.exp((GSI - 100) / 28)
    s = np.exp((GSI - 100) / 9)
    a = 0.5
    calculation_steps = [
        f"Geologic Strength Index (GSI): {GSI}",
        f"Material Constant (mb): {mb:.4f}",
        f"Parameter (s): {s:.4f}",
        f"Disturbance Factor (D): {disturbance_factor}",
        f"FS Calculation: FS = {rock_strength} / ({unit_weight} * {slope_height} * (mb * D * s) ** a)"
    ]
    FS = rock_strength / (unit_weight * slope_height * (mb * disturbance_factor * s) ** a)
    calculation_steps.append(f"Resulting FS: {FS:.4f}")
    return FS, calculation_steps

# Function to get method descriptions
def get_method_description(method):
    descriptions = {
        "Bishop": "The Bishop Method is an iterative, circular failure analysis method that approximates interslice forces, making it suitable for analyzing non-homogeneous slopes.",
        "Fellenius": "The Fellenius (Swedish Circle) Method is a simple circular method assuming no interslice forces, typically conservative and useful for homogeneous slopes.",
        "Janbu": "The Janbu Method is a slice-based, simplified method for non-circular failure surfaces, often used for slopes with complex geometries.",
        "Morgenstern-Price": "The Morgenstern-Price Method is a rigorous method that considers interslice forces, suitable for complex, non-circular failure surfaces.",
        "Taylor": "The Taylor's Method uses a stability number for homogeneous slopes, which is ideal for simple slopes with circular failure surfaces.",
        "Culmann": "Culmann's Method analyzes planar failure surfaces, commonly used for simple slopes where a planar failure is expected.",
        "Hoek-Brown": "The Hoek-Brown criterion is used specifically for rock slopes, applying empirical parameters based on rock mass strength."
    }
    return descriptions.get(method, "No description available.")

# Streamlit application
st.title("Slope Stability Analysis with Multiple Methods by Fabian J Manoppo Prompt with AI Tools")

# Input parameters
slope_height = st.number_input("Slope Height (m)", min_value=1.0, value=10.0)
slope_angle = np.radians(st.number_input("Slope Angle (degrees)", min_value=1.0, max_value=90.0, value=30.0))
cohesion = st.number_input("Cohesion (ton/m²)", min_value=0.0, value=3.2)
unit_weight = st.number_input("Unit Weight (ton/m³)", min_value=0.0, value=1.8)
friction_angle = np.radians(st.number_input("Friction Angle (degrees)", min_value=0.0, max_value=45.0, value=20.0))
num_slices = st.number_input("Number of Slices", min_value=1, max_value=50, value=10)
rock_strength = st.number_input("Rock Strength for Hoek-Brown (MPa)", min_value=0.0, value=10.0)
mi = st.number_input("Hoek-Brown Material Constant (mi)", min_value=0.0, value=10.0)
disturbance_factor = st.number_input("Hoek-Brown Disturbance Factor (D)", min_value=0.0, max_value=1.0, value=0.5)

# Select Method
method = st.selectbox("Select Method", ["Bishop", "Fellenius", "Janbu", "Morgenstern-Price", "Taylor", "Culmann", "Hoek-Brown"])

# Calculate Button
if st.button("Calculate"):
    # Display method description
    description = get_method_description(method)
    st.write(f"### {method} Method")
    st.write(description)
    
    if method == "Bishop":
        fs_bishop, calculation_table, calculation_steps = calculate_fs_bishop(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices)
        st.write(f"Factor of Safety (FS) using {method} Method: {fs_bishop:.3f}")
        st.write("Detailed Calculation Table for Bishop Method (Slice by Slice)")
        st.dataframe(calculation_table)
        st.write("### Calculation Steps")
        st.write("\n".join(calculation_steps))

    elif method == "Taylor":
        fs_taylor, calculation_steps = calculate_fs_taylor(cohesion, unit_weight, friction_angle, slope_height)
        st.write(f"Factor of Safety (FS) using {method} Method: {fs_taylor:.3f}")
        st.write("### Calculation Steps")
        st.write("\n".join(calculation_steps))

    elif method == "Culmann":
        fs_culmann, calculation_steps = calculate_fs_culmann(cohesion, unit_weight, friction_angle, slope_height, slope_angle)
        st.write(f"Factor of Safety (FS) using {method} Method: {fs_culmann:.3f}")
        st.write("### Calculation Steps")
        st.write("\n".join(calculation_steps))

    elif method == "Hoek-Brown":
        fs_hoek_brown, calculation_steps = calculate_fs_hoek_brown(rock_strength, mi, disturbance_factor, unit_weight, slope_height)
        st.write(f"Factor of Safety (FS) using {method} Method: {fs_hoek_brown:.3f}")
        st.write("### Calculation Steps")
        st.write("\n".join(calculation_steps))
