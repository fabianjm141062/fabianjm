import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define calculation functions for each method
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

# Additional Methods
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
    return FS, calculation_steps

def calculate_fs_spencer(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices):
    FS = (cohesion * slope_height) / (unit_weight * slope_height * np.tan(friction_angle))
    calculation_steps = [
        f"FS Calculation: FS = (Cohesion * Slope Height) / (Unit Weight * Slope Height * tan(Friction Angle))",
        f"FS = {FS:.4f}"
    ]
    return FS, calculation_steps

def calculate_fs_morgenstern_price(cohesion, unit_weight, friction_angle, slope_height, slope_angle):
    FS = cohesion / (unit_weight * slope_height * np.tan(friction_angle))
    calculation_steps = [
        f"FS = Cohesion / (Unit Weight * Slope Height * tan(Friction Angle))",
        f"FS = {FS:.4f}"
    ]
    return FS, calculation_steps

def calculate_fs_taylor(cohesion, unit_weight, slope_height):
    FS = cohesion / (unit_weight * slope_height)
    calculation_steps = [f"FS = Cohesion / (Unit Weight * Slope Height)", f"FS = {FS:.4f}"]
    return FS, calculation_steps

def calculate_fs_culmann(cohesion, unit_weight, friction_angle, slope_height, slope_angle):
    FS = cohesion / (unit_weight * slope_height * np.tan(friction_angle))
    calculation_steps = [f"FS = Cohesion / (Unit Weight * Slope Height * tan(Friction Angle))", f"FS = {FS:.4f}"]
    return FS, calculation_steps

def calculate_fs_hoek_brown(cohesion, unit_weight, friction_angle, slope_height, slope_angle):
    FS = (cohesion + unit_weight * slope_height * np.cos(friction_angle)) / (unit_weight * slope_height * np.sin(friction_angle))
    calculation_steps = [
        f"FS = (Cohesion + Unit Weight * Slope Height * cos(Friction Angle)) / (Unit Weight * Slope Height * sin(Friction Angle))",
        f"FS = {FS:.4f}"
    ]
    return FS, calculation_steps

# Streamlit Application
st.title("Slope Stability Analysis with Multiple Methods by Fabian J Manoppo prompt with AI Tools")

# Input parameters
slope_height = st.number_input("Slope Height (m)", min_value=1.0, value=10.0)
slope_angle = np.radians(st.number_input("Slope Angle (degrees)", min_value=1.0, max_value=90.0, value=30.0))
cohesion = st.number_input("Cohesion (ton/m²)", min_value=0.0, value=3.2)
unit_weight = st.number_input("Unit Weight (ton/m³)", min_value=0.0, value=1.8)
friction_angle = np.radians(st.number_input("Friction Angle (degrees)", min_value=0.0, max_value=45.0, value=20.0))
num_slices = st.number_input("Number of Slices", min_value=1, max_value=50, value=10)

# Select Method
method = st.selectbox("Select Method", ["Bishop", "Janbu", "Fellenius", "Spencer", "Morgenstern-Price", "Taylor", "Culmann", "Hoek-Brown"])

# Calculate Button
if st.button("Calculate"):
    st.write(f"### {method} Method")

    if method == "Bishop":
        fs_bishop, calculation_table, calculation_steps = calculate_fs_bishop(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices)
        st.write(f"Factor of Safety (FS): {fs_bishop:.3f}")
        st.dataframe(calculation_table)
        st.write("### Calculation Steps")
        st.write("\n".join(calculation_steps))
        
    elif method == "Janbu":
        fs_janbu, calculation_steps = calculate_fs_janbu(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices)
        st.write(f"Factor of Safety (FS): {fs_janbu:.3f}")
        st.write("### Calculation Steps")
        st.write("\n".join(calculation_steps))

    elif method == "Fellenius":
        fs_fellenius, calculation_steps = calculate_fs_fellenius(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices)
        st.write(f"Factor of Safety (FS): {fs_fellenius:.3f}")
        st.write("### Calculation Steps")
        st.write("\n".join(calculation_steps))

    elif method == "Spencer":
        fs_spencer, calculation_steps = calculate_fs_spencer(cohesion, unit_weight, friction_angle, slope_height, slope_angle, num_slices)
        st.write(f"Factor of Safety (FS): {fs_spencer:.3f}")
        st.write("### Calculation Steps")
        st.write("\n".join(calculation_steps))

    elif method == "Morgenstern-Price":
        fs_morgenstern_price, calculation_steps = calculate_fs_morgenstern_price(cohesion, unit_weight, friction_angle, slope_height, slope_angle)
             st.write(f"Factor of Safety (FS): {fs_morgenstern_price:.3f}")
        st.write("### Calculation Steps")
        st.write("\n".join(calculation_steps))

    elif method == "Taylor":
        fs_taylor, calculation_steps = calculate_fs_taylor(cohesion, unit_weight, slope_height)
        st.write(f"Factor of Safety (FS): {fs_taylor:.3f}")
        st.write("### Calculation Steps")
        st.write("\n".join(calculation_steps))

    elif method == "Culmann":
        fs_culmann, calculation_steps = calculate_fs_culmann(cohesion, unit_weight, friction_angle, slope_height, slope_angle)
        st.write(f"Factor of Safety (FS): {fs_culmann:.3f}")
        st.write("### Calculation Steps")
        st.write("\n".join(calculation_steps))

    elif method == "Hoek-Brown":
        fs_hoek_brown, calculation_steps = calculate_fs_hoek_brown(cohesion, unit_weight, friction_angle, slope_height, slope_angle)
        st.write(f"Factor of Safety (FS): {fs_hoek_brown:.3f}")
        st.write("### Calculation Steps")
        st.write("\n".join(calculation_steps))

    # Summary Table
    st.write("### Summary Table")
    summary_data = {
        "Method": ["Bishop", "Janbu", "Fellenius", "Spencer", "Morgenstern-Price", "Taylor", "Culmann", "Hoek-Brown"],
        "Factor of Safety": [
            fs_bishop if method == "Bishop" else None,
            fs_janbu if method == "Janbu" else None,
            fs_fellenius if method == "Fellenius" else None,
            fs_spencer if method == "Spencer" else None,
            fs_morgenstern_price if method == "Morgenstern-Price" else None,
            fs_taylor if method == "Taylor" else None,
            fs_culmann if method == "Culmann" else None,
            fs_hoek_brown if method == "Hoek-Brown" else None,
        ]
    }
    summary_df = pd.DataFrame(summary_data).dropna()
    st.dataframe(summary_df)

    # Diagram Placeholder for Selected Method
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label=f"{method} Method Diagram")
    ax.set_title(f"{method} Method Diagram")
    ax.legend()
    st.pyplot(fig)

# Descriptions of Each Method
method_descriptions = {
    "Bishop": "The Bishop Method approximates the factor of safety using an iterative process suitable for circular failure surfaces.",
    "Janbu": "The Janbu Method simplifies stability calculations for slopes with complex geometries, assuming plane failure surfaces.",
    "Fellenius": "The Fellenius (Swedish Circle) Method provides a simple approach to circular failure analysis, ideal for homogeneous slopes.",
    "Spencer": "The Spencer Method provides rigorous slope stability analysis by including both force and moment equilibrium.",
    "Morgenstern-Price": "The Morgenstern-Price Method is a robust method that accounts for interslice forces, ideal for complex non-circular failures.",
    "Taylor": "The Taylor Method uses stability charts and is particularly useful for rapid stability checks.",
    "Culmann": "The Culmann Method estimates stability by assuming a simple planar failure surface, typically used for rock slopes.",
    "Hoek-Brown": "The Hoek-Brown Method is specifically designed for rock slopes, providing an empirical stability estimation."
}

st.write("### Method Descriptions")
st.write(method_descriptions.get(method, "No description available for this method."))
 
