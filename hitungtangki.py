import streamlit as st
import pandas as pd
from PIL import Image

# Title of the app
st.title('Mechanical Calculation for RBDPO Tank')

# Section 1: Input Data Form
st.header("Input Data")

design_code = st.text_input("Design Code", "API 650 13th Edition Errata 1, 2021")
design_pressure = st.text_input("Design Pressure (MPag)", "Full of Water")
design_temp = st.number_input("Design Temperature (°C)", value=90)
inside_diameter = st.number_input("Inside Diameter (mm)", value=17000)
height = st.number_input("Height (mm)", value=16300)
specific_gravity = st.text_input("Design/Product Specific Gravity", "1.0 / 0.902 (RBDPO)")
roof_type = st.text_input("Roof Type", "Supported Fixed Cone Roof (1:12)")
joint_efficiency = st.text_input("Joint Efficiency (Shell/Roof/Bottom)", "0.85 / 0.85 / 0.85")
corrosion_shell = st.number_input("Corrosion Allowance - Shell (mm)", value=1.0)
corrosion_roof = st.number_input("Corrosion Allowance - Roof (mm)", value=1.0)
corrosion_structure = st.number_input("Corrosion Allowance - Structure (mm)", value=0.5)
wind_velocity = st.number_input("Wind Velocity (m/s)", value=39.1)
seismic_ss = st.number_input("Seismic Load (Ss)", value=0.115)
seismic_s1 = st.number_input("Seismic Load (S1)", value=0.082)
seismic_group = st.selectbox("Seismic Use Group", ["I", "II", "III"], index=1)
live_load = st.number_input("Live Load (kPa)", value=1.2)

# Section 2: Output Calculations
st.header("Output Summary")

# Example calculations for weights (these would be replaced with your actual calculations)
bottom_weight = 14_748  # in kg
shell_weight = 51_442   # in kg
roof_weight = 11_100    # in kg
empty_weight = bottom_weight + shell_weight + roof_weight
operating_weight = 3_432_883  # in kg
test_weight = 3_795_461  # in kg

# Example shell course thicknesses
shell_courses = {
    "Shell Course 1": "2500 x 10 mm",
    "Shell Course 2": "2500 x 10 mm",
    "Shell Course 3": "2500 x 8 mm",
    "Shell Course 4": "2500 x 6 mm",
    "Shell Course 5": "2300 x 6 mm",
    "Shell Course 6": "2000 x 6 mm",
    "Shell Course 7": "2000 x 6 mm"
}

# Display the calculated results
st.subheader("Weight Summary")
st.write(f"Bottom Weight: {bottom_weight} kg")
st.write(f"Shell Weight: {shell_weight} kg")
st.write(f"Roof Weight: {roof_weight} kg")
st.write(f"Empty Weight: {empty_weight} kg")
st.write(f"Operating Weight: {operating_weight} kg")
st.write(f"Test Weight: {test_weight} kg")

st.subheader("Shell Course Thickness Summary")
for course, thickness in shell_courses.items():
    st.write(f"{course}: {thickness}")

# Section 3: Display Design Image
st.subheader("Tank Design")

# Load the image from the directory or a URL
image = Image.open("tank_design.png")  # Replace with the path to your image
st.image(image, caption="Tank Design Schematic", use_column_width=True)

# Section 4: Option to download the results
st.subheader("Download Summary as Excel")

# Create a DataFrame for output data
output_data = {
    "Description": ["Bottom Weight", "Shell Weight", "Roof Weight", "Empty Weight", "Operating Weight", "Test Weight"],
    "Value": [f"{bottom_weight} kg", f"{shell_weight} kg", f"{roof_weight} kg", f"{empty_weight} kg", f"{operating_weight} kg", f"{test_weight} kg"]
}
output_df = pd.DataFrame(output_data)

# Provide a download button for the user to download the Excel file
@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')

csv = convert_df(output_df)

st.download_button(
    label="Download Output as CSV",
    data=csv,
    file_name='output_summary.csv',
    mime='text/csv',
)
