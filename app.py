import streamlit as st
import numpy as np
import plotly.graph_objects as go

# Function to simulate data
def generate_data(diameter, depth):
    # This function should be replaced with the actual simulation logic
    z = np.linspace(0, depth, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta, z = np.meshgrid(theta, z)
    r = diameter/2 + z*0  # Constant radius equal to half the diameter
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    return x, y, z

# Streamlit interface setup
st.title('3D Visualization of Casing Removal Effects')

# User inputs
diameter = st.slider('Column Diameter (m)', 0.5, 2.0, 1.0)
depth = st.slider('Column Depth (m)', 1.0, 10.0, 5.0)

# Button to perform simulation
if st.button('Run Simulation'):
    x, y, z = generate_data(diameter, depth)
    
    # Plotting
    fig = go.Figure(data=[go.Surface(z=z, x=x, y=y)])
    fig.update_layout(title='3D View of the Column', autosize=True,
                      scene=dict(
                          xaxis_title='X AXIS',
                          yaxis_title='Y AXIS',
                          zaxis_title='Z AXIS'))
    st.plotly_chart(fig, use_container_width=True)
