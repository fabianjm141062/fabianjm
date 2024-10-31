import streamlit as st
import pandas as pd
import numpy as np

class SheetPileStability:
    def __init__(self, active_moment, passive_moment, required_safety_factor=1.5):
        self.active_moment = active_moment
        self.passive_moment = passive_moment
        self.required_safety_factor = required_safety_factor

    def calculate_safety_factor(self):
        # Calculate the safety factor based on moments
        if self.active_moment != 0:
            safety_factor = abs(self.passive_moment / self.active_moment)
        else:
            safety_factor = 0  # To handle division by zero

        stability = "Safe" if safety_factor >= self.required_safety_factor else "Unsafe"
        return round(safety_factor, 2), stability

# Streamlit UI for input
st.title("Sheet Pile Stability Control")

# Input fields for active and passive moments
active_moment = st.number_input("Enter Total Active Moment (kNm):", min_value=0.0)
passive_moment = st.number_input("Enter Total Passive Moment (kNm):", min_value=0.0)

# Create the stability checker
stability_checker = SheetPileStability(active_moment, passive_moment)

# Calculate and display results when the button is clicked
if st.button("Check Stability"):
    safety_factor, stability = stability_checker.calculate_safety_factor()
    
    # Display the results in a DataFrame format for clarity
    results_df = pd.DataFrame({
        'Total Active Moment (kNm)': [active_moment],
        'Total Passive Moment (kNm)': [passive_moment],
        'Safety Factor': [safety_factor],
        'Stability': [stability]
    })
    
    st.subheader("Stability Analysis Results")
    st.dataframe(results_df)
