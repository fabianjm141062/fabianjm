import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Bishop Method Calculation
def calculate_fs_bishop(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices, tolerance=0.001, max_iterations=100):
    FS = 1.0
    slice_width = slope_height / num_slices
    R = slope_height
    calculation_steps = []

    for iteration in range(max_iterations):
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
        calculation_steps.append([iteration + 1, new_FS, numerator_sum, denominator_sum])
        if abs(new_FS - FS) < tolerance:
            FS = new_FS
            break
        FS = new_FS

    steps_df = pd.DataFrame(calculation_steps, columns=["Iteration", "FS", "Numerator Sum", "Denominator Sum"])
    return FS, steps_df

# Janbu Method Calculation
def calculate_fs_janbu(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices):
    slice_width = slope_height / num_slices
    calculation_steps = []
    numerator_sum = 0
    denominator_sum = 0

    for i in range(num_slices):
        x = (i + 0.5) * slice_width
        height_slice = slope_height - x * np.tan(slope_angle)
        weight = unit_weight * slice_width * height_slice
        normal_force = weight * np.cos(slope_angle)
        shear_resistance = cohesion * slice_width + normal_force * np.tan(friction_angle)
        numerator_sum += shear_resistance
        denominator_sum += weight * np.sin(slope_angle)
        calculation_steps.append([i + 1, weight, normal_force, shear_resistance])
    
    FS = numerator_sum / denominator_sum
    steps_df = pd.DataFrame(calculation_steps, columns=["Slice", "Weight", "Normal Force", "Shear Resistance"])
    return FS, steps_df

# Fellenius Method Calculation
def calculate_fs_fellenius(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices):
    slice_width = slope_height / num_slices
    calculation_steps = []
    numerator_sum = 0
    denominator_sum = 0

    for i in range(num_slices):
        x = (i + 0.5) * slice_width
        theta = np.arctan(slice_width / slope_height)
        height_slice = slope_height - x * np.tan(slope_angle)
        weight = unit_weight * slice_width * height_slice
        normal_force = weight * np.cos(theta)
        shear_resistance = cohesion * slice_width + normal_force * np.tan(friction_angle)
        numerator_sum += shear_resistance
        denominator_sum += weight * np.sin(theta)
        calculation_steps.append([i + 1, weight, normal_force, shear_resistance])

    FS = numerator_sum / denominator_sum
    steps_df = pd.DataFrame(calculation_steps, columns=["Slice", "Weight", "Normal Force", "Shear Resistance"])
    return FS, steps_df

# Spencer Method Calculation
def calculate_fs_spencer(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices):
    FS = (cohesion * slope_height) / (unit_weight * slope_height * np.tan(friction_angle))
    steps_df = pd.DataFrame([[1, FS]], columns=["Step", "FS"])
    return FS, steps_df

# Morgenstern-Price Method Calculation
def calculate_fs_morgenstern_price(cohesion, unit_weight, friction_angle, slope_height, slope_angle):
    FS = cohesion / (unit_weight * slope_height * np.tan(friction_angle))
    steps_df = pd.DataFrame([[1, FS]], columns=["Step", "FS"])
    return FS, steps_df

# Taylor's Method Calculation
def calculate_fs_taylor(cohesion, unit_weight, slope_height):
    N = 0.261  # Example stability number
    FS = cohesion / (unit_weight * slope_height * N)
    steps_df = pd.DataFrame([[1, N, FS]], columns=["Step", "Stability Number (N)", "FS"])
    return FS, steps_df

# Culmann's Method Calculation
def calculate_fs_culmann(cohesion, unit_weight, friction_angle, slope_height, slope_angle):
    critical_angle = np.degrees(slope_angle) - np.degrees(friction_angle) / 2
    FS = cohesion / (unit_weight * slope_height * np.sin(np.radians(critical_angle)) * np.cos(np.radians(critical_angle)))
    steps_df = pd.DataFrame([[1, critical_angle, FS]], columns=["Step", "Critical Angle", "FS"])
    return FS, steps_df

# Hoek-Brown Method Calculation
def calculate_fs_hoek_brown(rock_strength, mi, disturbance_factor, unit_weight, slope_height):
    GSI = 60  # Example Geologic Strength Index (GSI) value
    mb = mi * np.exp((GSI - 100) / 28)
    s = np.exp((GSI - 100) / 9)
    a = 0.5
    FS = rock_strength / (unit_weight * slope_height * (mb * disturbance_factor * s) ** a)
    steps_df = pd.DataFrame([[1, GSI, mb, s, FS]], columns=["Step", "GSI", "mb", "s", "FS"])
    return FS, steps_df

# Streamlit Application
st.title("Slope Stability Analysis with Multiple Methods by Fabian J Manoppo with AI Tools")

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

    # Perform calculation based on selected method
    if method == "Bishop":
        fs, steps_df = calculate_fs_bishop(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices)
    elif method == "Janbu":
        fs, steps_df = calculate_fs_janbu(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices)
    elif method == "Fellenius":
        fs, steps_df = calculate_fs_fellenius(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices)
    elif method == "Spencer":
        fs, steps_df = calculate_fs_spencer(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices)
    elif method == "Morgenstern-Price":
        fs, steps_df = calculate_fs_morgenstern_price(cohesion, unit_weight, friction_angle, slope_height, slope_angle)
    elif method == "Taylor":
        fs, steps_df = calculate_fs_taylor(cohesion, unit_weight, slope_height)
    elif method == "Culmann":
        fs, steps_df = calculate_fs_culmann(cohesion, unit_weight, friction_angle, slope_height, slope_angle)
    elif method == "Hoek-Brown":
        fs, steps_df = calculate_fs_hoek_brown(rock_strength, mi, disturbance_factor, unit_weight, slope_height)

    # Display Factor of Safety
    st.write(f"Factor of Safety (FS): {fs:.3f}")

    # Display Computation Steps in a Table
    st.write("### Detailed Computation Steps")
    st.dataframe(steps_df)

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

    # Display the plot for the selected method
    plot_slope(method, fs, slope_height, slope_angle)

# Descriptions of each method
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

# Display the selected method's description
st.write("### Method Description and Use Case")
st.write(method_descriptions.get(method, "No description available for this method."))
