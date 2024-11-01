import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Bishop Method Calculation
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

# Taylor's Method Calculation
def calculate_fs_taylor(cohesion, unit_weight, slope_height):
    # Stability number, N, is derived from empirical stability charts. Using an example value.
    N = 0.261  # Example stability number for a given slope angle and friction angle
    FS = cohesion / (unit_weight * slope_height * N)
    calculation_steps = [
        "Using Taylor's Stability Charts method",
        f"Stability Number (N): {N}",
        f"FS = Cohesion / (Unit Weight * Slope Height * N)",
        f"FS = {FS:.4f}"
    ]
    return FS, calculation_steps

# Culmann's Method Calculation
def calculate_fs_culmann(cohesion, unit_weight, friction_angle, slope_height, slope_angle):
    critical_angle = np.degrees(slope_angle) - np.degrees(friction_angle) / 2  # Approximate critical failure angle
    FS = cohesion / (unit_weight * slope_height * np.sin(np.radians(critical_angle)) * np.cos(np.radians(critical_angle)))
    calculation_steps = [
        "Using Culmann's planar failure method",
        f"Critical Failure Angle (degrees): {critical_angle:.2f}",
        f"FS = Cohesion / (Unit Weight * Slope Height * sin(critical_angle) * cos(critical_angle))",
        f"FS = {FS:.4f}"
    ]
    return FS, calculation_steps

# Hoek-Brown Method Calculation
def calculate_fs_hoek_brown(rock_strength, mi, disturbance_factor, unit_weight, slope_height):
    GSI = 60  # Example Geologic Strength Index (GSI) value
    mb = mi * np.exp((GSI - 100) / 28)
    s = np.exp((GSI - 100) / 9)
    a = 0.5
    FS = rock_strength / (unit_weight * slope_height * (mb * disturbance_factor * s) ** a)
    calculation_steps = [
        f"Geologic Strength Index (GSI): {GSI}",
        f"Material Constant (mb): {mb:.4f}",
        f"Parameter (s): {s:.4f}",
        f"Disturbance Factor (D): {disturbance_factor}",
        f"FS Calculation: FS = {rock_strength} / ({unit_weight} * {slope_height} * (mb * D * s) ** a)",
        f"Resulting FS: {FS:.4f}"
    ]
    return FS, calculation_steps

# Plotting function
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

# Streamlit Application
st.title("Slope Stability Analysis with Multiple Methods")

# Select Method
method = st.selectbox("Select Method", ["Bishop", "Taylor", "Culmann", "Hoek-Brown"])

# Input Parameters Based on Method
if method == "Bishop":
    slope_height = st.number_input("Slope Height (m)", min_value=1.0, value=10.0)
    slope_angle = np.radians(st.number_input("Slope Angle (degrees)", min_value=1.0, max_value=90.0, value=30.0))
    cohesion = st.number_input("Cohesion (ton/m²)", min_value=0.0, value=3.2)
    unit_weight = st.number_input("Unit Weight (ton/m³)", min_value=0.0, value=1.8)
    friction_angle = np.radians(st.number_input("Friction Angle (degrees)", min_value=0.0, max_value=45.0, value=20.0))
    num_slices = st.number_input("Number of Slices", min_value=1, max_value=50, value=10)

elif method == "Taylor":
    cohesion = st.number_input("Cohesion (ton/m²)", min_value=0.0, value=3.2)
    unit_weight = st.number_input("Unit Weight (ton/m³)", min_value=0.0, value=1.8)
    slope_height = st.number_input("Slope Height (m)", min_value=1.0, value=10.0)

elif method == "Culmann":
    cohesion = st.number_input("Cohesion (ton/m²)", min_value=0.0, value=3.2)
    unit_weight = st.number_input("Unit Weight (ton/m³)", min_value=0.0, value=1.8)
    friction_angle = np.radians(st.number_input("Friction Angle (degrees)", min_value=0.0, max_value=45.0, value=20.0))
    slope_height = st.number_input("Slope Height (m)", min_value=1.0, value=10.0)
    slope_angle = np.radians(st.number_input("Slope Angle (degrees)", min_value=1.0, max_value=90.0, value=30.0))

elif method == "Hoek-Brown":
    rock_strength = st.number_input("Rock Mass Strength (MPa)", min_value=0.0, value=10.0)
    mi = st.number_input("Material Constant (mi)", min_value=0.0, value=10.0)
    disturbance_factor = st.number_input("Disturbance Factor (D)", min_value=0.0, max_value=1.0, value=0.5)
    unit_weight = st.number_input("Unit Weight (ton/m³)", min_value=0.0, value=1.8)
    slope_height = st.number_input("Slope Height (m)", min_value=1.0, value=10.0)

# Calculate Button
if st.button("Calculate"):
    st.write(f"### {method} Method")

    if method == "Bishop":
        fs_bishop, calculation_table, calculation_steps = calculate_fs_bishop(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices)
        st.write(f"Factor of Safety (FS): {fs_bishop:.3f}")
        st.dataframe(calculation_table)
        st.write("### Calculation Steps")
        st.write("\n".join(calculation_steps))
        import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Bishop Method Calculation
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

# Taylor's Method Calculation
def calculate_fs_taylor(cohesion, unit_weight, slope_height):
    # Stability number, N, is derived from empirical stability charts. Using an example value.
    N = 0.261  # Example stability number for a given slope angle and friction angle
    FS = cohesion / (unit_weight * slope_height * N)
    calculation_steps = [
        "Using Taylor's Stability Charts method",
        f"Stability Number (N): {N}",
        f"FS = Cohesion / (Unit Weight * Slope Height * N)",
        f"FS = {FS:.4f}"
    ]
    return FS, calculation_steps

# Culmann's Method Calculation
def calculate_fs_culmann(cohesion, unit_weight, friction_angle, slope_height, slope_angle):
    critical_angle = np.degrees(slope_angle) - np.degrees(friction_angle) / 2  # Approximate critical failure angle
    FS = cohesion / (unit_weight * slope_height * np.sin(np.radians(critical_angle)) * np.cos(np.radians(critical_angle)))
    calculation_steps = [
        "Using Culmann's planar failure method",
        f"Critical Failure Angle (degrees): {critical_angle:.2f}",
        f"FS = Cohesion / (Unit Weight * Slope Height * sin(critical_angle) * cos(critical_angle))",
        f"FS = {FS:.4f}"
    ]
    return FS, calculation_steps

# Hoek-Brown Method Calculation
def calculate_fs_hoek_brown(rock_strength, mi, disturbance_factor, unit_weight, slope_height):
    GSI = 60  # Example Geologic Strength Index (GSI) value
    mb = mi * np.exp((GSI - 100) / 28)
    s = np.exp((GSI - 100) / 9)
    a = 0.5
    FS = rock_strength / (unit_weight * slope_height * (mb * disturbance_factor * s) ** a)
    calculation_steps = [
        f"Geologic Strength Index (GSI): {GSI}",
        f"Material Constant (mb): {mb:.4f}",
        f"Parameter (s): {s:.4f}",
        f"Disturbance Factor (D): {disturbance_factor}",
        f"FS Calculation: FS = {rock_strength} / ({unit_weight} * {slope_height} * (mb * D * s) ** a)",
        f"Resulting FS: {FS:.4f}"
    ]
    return FS, calculation_steps

# Plotting function
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

# Streamlit Application
st.title("Slope Stability Analysis with Multiple Methods by Fabian J Manoppo with AI Tools")

# Select Method
method = st.selectbox("Select Method", ["Bishop", "Taylor", "Culmann", "Hoek-Brown"])

# Input Parameters Based on Method
if method == "Bishop":
    slope_height = st.number_input("Slope Height (m)", min_value=1.0, value=10.0)
    slope_angle = np.radians(st.number_input("Slope Angle (degrees)", min_value=1.0, max_value=90.0, value=30.0))
    cohesion = st.number_input("Cohesion (ton/m²)", min_value=0.0, value=3.2)
    unit_weight = st.number_input("Unit Weight (ton/m³)", min_value=0.0, value=1.8)
    friction_angle = np.radians(st.number_input("Friction Angle (degrees)", min_value=0.0, max_value=45.0, value=20.0))
    num_slices = st.number_input("Number of Slices", min_value=1, max_value=50, value=10)

elif method == "Taylor":
    cohesion = st.number_input("Cohesion (ton/m²)", min_value=0.0, value=3.2)
    unit_weight = st.number_input("Unit Weight (ton/m³)", min_value=0.0, value=1.8)
    slope_height = st.number_input("Slope Height (m)", min_value=1.0, value=10.0)

elif method == "Culmann":
    cohesion = st.number_input("Cohesion (ton/m²)", min_value=0.0, value=3.2)
    unit_weight = st.number_input("Unit Weight (ton/m³)", min_value=0.0, value=1.8)
    friction_angle = np.radians(st.number_input("Friction Angle (degrees)", min_value=0.0, max_value=45.0, value=20.0))
    slope_height = st.number_input("Slope Height (m)", min_value=1.0, value=10.0)
    slope_angle = np.radians(st.number_input("Slope Angle (degrees)", min_value=1.0, max_value=90.0, value=30.0))

elif method == "Hoek-Brown":
    rock_strength = st.number_input("Rock Mass Strength (MPa)", min_value=0.0, value=10.0)
    mi = st.number_input("Material Constant (mi)", min_value=0.0, value=10.0)
    disturbance_factor = st.number_input("Disturbance Factor (D)", min_value=0.0, max_value=1.0, value=0.5)
    unit_weight = st.number_input("Unit Weight (ton/m³)", min_value=0.0, value=1.8)
    slope_height = st.number_input("Slope Height (m)", min_value=1.0, value=10.0)

# Calculate Button
if st.button("Calculate"):
    st.write(f"### {method} Method")

    if method == "Bishop":
        fs_bishop, calculation_table, calculation_steps = calculate_fs_bishop(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices)
        st.write(f"Factor of Safety (FS): {fs_bishop:.3f}")
        st.dataframe(calculation_table)
        st.write("### Calculation Steps")
        st.write("\n".join(calculation_steps))
        plot_slope("Bishop", fs_bishop, slope_height, slope_angle)

    elif method == "Taylor":
        fs_taylor, calculation_steps = calculate_fs_taylor(cohesion, unit_weight, slope_height)
        st.write(f"Factor of Safety (FS): {fs_taylor:.3f}")
        st.write("### Calculation Steps")
        st.write("\n".join(calculation_steps))
        plot_slope("Taylor", fs_taylor, slope_height, np.radians(45))  # Using a default angle for Taylor's method

    elif method == "Culmann":
        fs_culmann, calculation_steps = calculate_fs_culmann(cohesion, unit_weight, friction_angle, slope_height, slope_angle)
        st.write(f"Factor of Safety (FS): {fs_culmann:.3f}")
        st.write("### Calculation Steps")
        st.write("\n".join(calculation_steps))
        plot_slope("Culmann", fs_culmann, slope_height, slope_angle)

    elif method == "Hoek-Brown":
        fs_hoek_brown, calculation_steps = calculate_fs_hoek_brown(rock_strength, mi, disturbance_factor, unit_weight, slope_height)
        st.write(f"Factor of Safety (FS): {fs_hoek_brown:.3f}")
        st.write("### Calculation Steps")
        st.write("\n".join(calculation_steps))
        plot_slope("Hoek-Brown", fs_hoek_brown, slope_height, np.radians(45))  # Using a default angle for plotting

# Method Descriptions and Computation Explanation
method_descriptions = {
    "Bishop": "The Bishop Method is an iterative, slice-based method for circular failure surfaces. It considers vertical interslice forces and calculates FS by dividing total resisting moments by driving moments.",
    "Taylor": "Taylor's Method uses empirical stability charts for circular failure surfaces in homogeneous slopes. It calculates FS based on a stability number derived from the charts.",
    "Culmann": "The Culmann Method assumes a planar failure surface, commonly used for rock slopes where planar failures are expected. FS is calculated by analyzing the forces along the planar surface.",
    "Hoek-Brown": "The Hoek-Brown criterion is tailored for rock slopes and calculates FS based on empirical rock mass strength parameters. This method is ideal for slopes in rock masses with defined properties."
}

# Display the selected method's description and computation explanation
st.write("### Method Description and Computation Explanation")
st.write(method_descriptions.get(method, "No description available for this method."))


