import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Title of the app
st.title('Gravity Model for Trip Distribution oleh Fabian J Manoppo')

# Input data
st.header('Input Data')

T = np.array([
    [0, 50, 20, 10, 5],
    [50, 0, 30, 15, 10],
    [20, 30, 0, 25, 15],
    [10, 15, 25, 0, 20],
    [5, 10, 15, 20, 0]
])

P = np.array([1000, 800, 600, 400, 200])
E = np.array([500, 400, 300, 200, 100])

D = np.array([
    [0, 1, 2, 3, 4],
    [1, 0, 1, 2, 3],
    [2, 1, 0, 1, 2],
    [3, 2, 1, 0, 1],
    [4, 3, 2, 1, 0]
])

alpha = 1.0
beta = 1.0
gamma = 2.0

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
T_model = gravity_model(P, E, D, alpha, beta, gamma)
T_model_df = pd.DataFrame(T_model, columns=[f'Zona {i+1}' for i in range(5)], index=[f'Zona {i+1}' for i in range(5)])

# Display the matrix
st.subheader('Gravity Model Matrix')
st.dataframe(T_model_df)

# Plotting the heatmap
st.subheader('Heatmap of Trip Distribution')
plt.figure(figsize=(10, 8))
sns.heatmap(T_model_df, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=.5)
st.pyplot(plt)
