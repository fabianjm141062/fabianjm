import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Calculation functions for each method
def calculate_fs_bishop(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices, tolerance=0.001, max_iterations=100):
    FS = 1.0
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

# Janbu Method Calculation
def calculate_fs_janbu(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices):
    slice_width = slope_height / num_slices
    numerator_sum = 0
    denominator_sum = 0
    calculation_steps = []

    for i in range(num_slices):
        x = (i + 0.5) * slice_width
        height_slice = slope_height - x * np.tan(slope_angle)
        weight = unit_weight * slice_width * height_slice
        normal_force = weight * np.cos(slope_angle)
        shear_resistance = cohesion * slice_width + normal_force * np.tan(friction_angle)
        numerator_sum += shear_resistance
        denominator_sum += weight * np.sin(slope_angle)
        calculation_steps.append(f"Slice {i+1}: Weight = {weight:.2f}, Shear Resistance = {shear_resistance:.2f}, Normal Force = {normal_force:.2f}")
    
    FS = numerator_sum / denominator_sum
    return FS, calculation_steps

# Fellenius Method Calculation
def calculate_fs_fellenius(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices):
    slice_width = slope_height / num_slices
    numerator_sum = 0
    denominator_sum = 0
    calculation_steps = []

    for i in range(num_slices):
        x = (i + 0.5) * slice_width
        theta = np.arctan(slice_width / slope_height)
        height_slice = slope_height - x * np.tan(slope_angle)
        weight = unit_weight * slice_width * height_slice
        normal_force = weight * np.cos(theta)
        shear_resistance = cohesion * slice_width + normal_force * np.tan(friction_angle)
        numerator_sum += shear_resistance
        denominator_sum += weight * np.sin(theta)
        calculation_steps.append(f"Slice {i+1}: Weight = {weight:.2f}, Shear Resistance = {shear_resistance:.2f}, Normal Force = {normal_force:.2f}")

    FS = numerator_sum / denominator_sum
    return FS, calculation_steps

# Spencer Method Calculation
def calculate_fs_spencer(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices):
    FS = (cohesion * slope_height) / (unit_weight * slope_height * np.tan(friction_angle))
    calculation_steps = [
        f"FS Calculation: FS = (Cohesion * Slope Height) / (Unit Weight * Slope Height * tan(Friction Angle))",
        f"FS = {FS:.4f}"
    ]
    return FS, calculation_steps

# Morgenstern-Price Method Calculation
def calculate_fs_morgenstern_price(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices):
    FS = cohesion / (unit_weight * slope_height * np.tan(friction_angle))
    calculation_steps = [
        f"FS = Cohesion / (Unit Weight * Slope Height * tan(Friction Angle))",
        f"FS = {FS:.4f}"
    ]
    return FS, calculation_steps

# Taylor, Culmann, and Hoek-Brown functions (as previously defined)
# (Taylor, Culmann, and Hoek-Brown functions are not repeated here for brevity)

# Streamlit Application
st.title("Slope Stability Analysis with Multiple Methods")

# Select Method
method = st.selectbox("Select Method", ["Bishop", "Janbu", "Fellenius", "Spencer", "Morgenstern-Price", "Taylor", "Culmann", "Hoek-Brown"])

# Input Parameters
slope_height = st.number_input("Slope Height (m)", min_value=1.0, value=10.0)
slope_angle = np.radians(st.number_input("Slope Angle (degrees)", min_value=1.0, max_value=90.0, value=30.0))
cohesion = st.number_input("Cohesion (ton/m²)", min_value=0.0, value=3.2)
unit_weight = st.number_input("Unit Weight (ton/m³)", min_value=0.0, value=1.8)
friction_angle = np.radians(st.number_input("Friction Angle (degrees)", min_value=0.0, max_value=45.0, value=20.0))
num_slices = st.number_input("Number of Slices", min_value=1, max_value=50, value=10)

if method == "Hoek-Brown":
    rock_strength = st.number_input("Rock Mass Strength (MPa)", min_value=0.0, value=10.0)
    mi = st.number_input("Material Constant (mi)", min_value=0.0, value=10.0)
    disturbance_factor = st.number_input("Disturbance Factor (D)", min_value=0.0, max_value=1.0, value=0.5)

# Calculate Button
if st.button("Calculate"):
    st.write(f"### {method} Method")

    if method == "Bishop":
        fs, table, steps = calculate_fs_bishop(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices)
    elif method == "Janbu":
        fs, steps = calculate_fs_janbu(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices)
    elif method == "Fellenius":
        fs, steps = calculate_fs_fellenius(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices)
    elif method == "Spencer":
        fs, steps = calculate_fs_spencer(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices)
    elif method == "Morgenstern-Price":
        fs, steps = calculate_fs_morgenstern_price(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices)
    elif method == "Taylor":
        fs, steps = calculate_fs_taylor(cohesion, unit_weight, slope_height)
    elif method == "Culmann":
        fs, steps = calculate_fs_culmann(cohesion, unit_weight, friction_angle, slope_height, slope_angle)
    elif method == "Hoek-Brown":
        fs, steps = calculate_fs_hoek_brown(rock_strength, mi, disturbance_factor, unit_weight, slope_height)

    # Display results
    st.write(f"Factor of Safety (FS): {fs:.3f}")
    st.write("### Calculation Steps")
    st.write("\n".join(steps))
    
    # Plotting function for slope and failure surface
    def plot_slope(method_name, FS, slope_height, slope_angle):
        slope_width = slope_height / np.tan(slope_angle)
        x_slope = [0, slope_width]
        y_slope = [0, slope_height]

        R = slope_height if method_name in ["Bishop", "Taylor"] else slope_height * 1.2
        center_x = slope_width / 2
        center_y = slope_height - R

        theta = np.linspace(0, np.pi, 100)
        x_circle = center_x + R * np.cos(theta)
        y_circle = center_y + R * np.sin(theta)

        plt.figure(figsize=(8, 6))
        plt.plot(x_slope, y_slope, color='black', linewidth=2, label='Slope Surface')
        plt.plot(x_circle, y_circle, color='red', linestyle='--', label=f'{method_name} Failure Surface')
        plt.title(f"Slope Stability Analysis ({method_name} Method) - FS: {FS:.3f}")
        plt.xlabel("Width (m)")
        plt.ylabel("Height (m)")
        plt.legend()
        plt.grid()
        st.pyplot(plt)

    # Call the plotting function for the selected method
    plot_slope(method, fs, slope_height, slope_angle)

# Descriptions of Methods
method_descriptions = {
    "Bishop": "Iterative, slice-based; circular failure. Vertical interslice forces. Suitable for circular failures in non-homogeneous soil.",
    "Janbu": "Non-circular surfaces, balances vertical/horizontal forces. Suitable for complex geometries.",
    "Fellenius": "Simple, conservative circular method without interslice forces. Suitable for homogeneous slopes with circular failure.",
    "Spencer": "Rigorous, balances force/moment equilibrium; suitable for complex slopes with interslice forces.",
    "Morgenstern-Price": "Similar to Spencer, but flexible interslice force assumptions. Suitable for complex slopes.",
    "Taylor": "Uses stability charts for empirical FS estimation. Quick assessment for simple homogeneous slopes.",
    "Culmann": "Analyzes planar failure surfaces, suitable for rock slopes with planar failure.",
    "Hoek-Brown": "Empirical for rock masses, based on geologic strength. Suitable for rock slopes with significant rock masses."
}

# Display selected method's description and computation explanation
st.write("### Method Description and Use Case")
st.write(method_descriptions.get(method, "No description available for this method."))

