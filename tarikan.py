import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Title of the app
st.title('Gravity Model for Trip Distribution')

# Input section for user-provided data
st.header('Input Data')

# Input Population (P)
st.subheader('Population in each zone')
P = []
for i in range(1, 6):
    P.append(st.number_input(f'Population in Zone {i}', min_value=100, max_value=10000, value=1000, step=100))

# Input Employment/Attraction (E)
st.subheader('Employment/Attraction in each zone')
E = []
for i in range(1, 6):
    E.append(st.number_input(f'Employment in Zone {i}', min_value=50, max_value=5000, value=500, step=50))

# Input Distance Matrix (D)
st.subheader('Distance between zones (provide symmetric distance matrix)')
D = np.zeros((5, 5))
for i in range(5):
    for j in range(5):
        if i != j:
            D[i][j] = st.number_input(f'Distance from Zone {i+1} to Zone {j+1}', min_value=0.1, max_value=100.0, value=1.0, step=0.1)

# Gravity Model Parameters
st.subheader('Gravity Model Parameters')
alpha = st.number_input('Alpha (constant)', min_value=0.0, value=1.0)
beta = st.number_input('Beta (population exponent)', min_value=0.0, value=1.0)
gamma = st.number_input('Gamma (employment exponent)', min_value=0.0, value=2.0)

# Function to calculate gravity model
def gravity_model(P, E, D, alpha, beta, gamma):
    n = len(P)
    T_model = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                T_model[i][j] = alpha * (P[i]**beta * E[j]**gamma) / D[i][j]**2
            else:
                T_model[i][j] = 0
    return T_model

# Calculate the model
P = np.array(P)
E = np.array(E)
T_model = gravity_model(P, E, D, alpha, beta, gamma)
T_model_df = pd.DataFrame(T_model, columns=[f'Zone {i+1}' for i in range(5)], index=[f'Zone {i+1}' for i in range(5)])

# Display the matrix
st.subheader('Gravity Model Matrix')
st.dataframe(T_model_df)

# Plotting the heatmap
st.subheader('Heatmap of Trip Distribution')
plt.figure(figsize=(10, 8))
sns.heatmap(T_model_df, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=.5)
st.pyplot(plt)
