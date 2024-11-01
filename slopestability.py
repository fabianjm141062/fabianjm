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

# Fellenius (Swedish Circle) Method
def calculate_fs_fellenius(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices):
    slice_width = slope_height / num_slices
    R = slope_height
    numerator_sum = 0
    denominator_sum = 0
    calculation_steps = []
    
    for i in range(num_slices):
        x = (i + 0.5) * slice_width
        theta = np.arctan(slice_width / R)
        height_slice = slope_height - x * np.tan(slope_angle)
        weight = unit_weight * slice_width * height_slice
        normal_force = weight * np.cos(theta)
        shear_resistance = cohesion * slice_width + normal_force * np.tan(friction_angle)
        numerator_sum += shear_resistance
        denominator_sum += weight * np.sin(theta)
        calculation_steps.append(f"Slice {i+1}: Weight = {weight:.2f}, Shear Resistance = {shear_resistance:.2f}, Normal Force = {normal_force:.2f}")
    
    FS = numerator_sum / denominator_sum
    calculation_steps.append(f"Final FS = {FS:.4f} (Numerator Sum = {numerator_sum:.4f}, Denominator Sum = {denominator_sum:.4f})")
    return FS, calculation_steps

# Janbu Method (Simplified)
def calculate_fs_janbu(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices):
    slice_width = slope_height / num_slices
    R = slope_height
    numerator_sum = 0
    denominator_sum = 0
    calculation_steps = []
    
    for i in range(num_slices):
        x = (i + 0.5) * slice_width
        theta = np.arctan(slice_width / R)
        height_slice = slope_height - x * np.tan(slope_angle)
        weight = unit_weight * slice_width * height_slice
        normal_force = weight * np.cos(theta) / np.cos(theta)
        shear_resistance = cohesion * slice_width + normal_force * np.tan(friction_angle)
        numerator_sum += shear_resistance
        denominator_sum += weight * np.sin(theta)
        calculation_steps.append(f"Slice {i+1}: Weight = {weight:.2f}, Shear Resistance = {shear_resistance:.2f}, Normal Force = {normal_force:.2f}")
    
    FS = numerator_sum / denominator_sum
    calculation_steps.append(f"Final FS = {FS:.4f} (Numerator Sum = {numerator_sum:.4f}, Denominator Sum = {denominator_sum:.4f})")
    return FS, calculation_steps

# Morgenstern-Price Method (Placeholder example)
def calculate_fs_morgenstern_price(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices):
    FS = (cohesion * slope_height) / (unit_weight * slope_height * np.tan(friction_angle))
    calculation_steps = [
        f"Assumed FS Calculation: FS = (Cohesion * Slope Height) / (Unit Weight * Slope Height * tan(Friction Angle))",
        f"FS = {FS:.4f}"
    ]
    return FS, calculation_steps

# Function to get method descriptions
def get_method_description(method):
    descriptions = {
        "Bishop": "The Bishop Method is an iterative, circular failure analysis method that approximates interslice forces, making it suitable for analyzing non-homogeneous slopes.",
        "Fellenius": "The Fellenius (Swedish Circle) Method is a simple circular method assuming no interslice forces, typically conservative and useful for homogeneous slopes.",
        "Janbu": "The Janbu Method is a slice-based, simplified method for non-circular failure surfaces, often used for slopes with complex geometries.",
        "Morgenstern-Price": "The Morgenstern-Price Method is a rigorous method that considers interslice forces, suitable for complex, non-circular failure surfaces."
    }
    return descriptions.get(method, "No description available.")

# Streamlit application
st.title("Slope Stability Analysis with Multiple Methods by Fabian J Manoppo Prompt With AI Tools")

# Input parameters
slope_height = st.number_input("Slope Height (m)", min_value=1.0, value=10.0)
slope_angle = np.radians(st.number_input("Slope Angle (degrees)", min_value=1.0, max_value=90.0, value=30.0))
cohesion = st.number_input("Cohesion (ton/m²)", min_value=0.0, value=3.2)
unit_weight = st.number_input("Unit Weight (ton/m³)", min_value=0.0, value=1.8)
friction_angle = np.radians(st.number_input("Friction Angle (degrees)", min_value=0.0, max_value=45.0, value=20.0))
num_slices = st.number_input("Number of Slices", min_value=1, max_value=50, value=10)

# Select Method
method = st.selectbox("Select Method", ["Bishop", "Fellenius", "Janbu", "Morgenstern-Price"])

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

    elif method == "Fellenius":
        fs_fellenius, calculation_steps = calculate_fs_fellenius(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices)
        st.write(f"Factor of Safety (FS) using {method} Method: {fs_fellenius:.3f}")
        st.write("### Calculation Steps")
        st.write("\n".join(calculation_steps))

    elif method == "Janbu":
        fs_janbu, calculation_steps = calculate_fs_janbu(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices)
        st.write(f"Factor of Safety (FS) using {method} Method: {fs_janbu:.3f}")
        st.write("### Calculation Steps")
        st.write("\n".join(calculation_steps))

    elif method == "Morgenstern-Price":
        fs_morgenstern_price, calculation_steps = calculate_fs_morgenstern_price(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices)
        st.write(f"Factor of Safety (FS) using {method} Method: {fs_morgenstern_price:.3f}")
        st.write("### Calculation Steps")
        st.write("\n".join(calculation_steps))
