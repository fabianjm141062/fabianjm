import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the dataset (with error handling)
@st.cache
def load_data():
    try:
        # Assuming the dataset is named 'hypertension_data.csv'
        data = pd.read_csv('hypertension_data.csv')
        return data
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Function to display information about each variable
def display_variable_info(data, features):
    st.write("### Variable Information")
    for feature in features:
        min_val = data[feature].min()
        max_val = data[feature].max()
        st.write(f"- **{feature}**: Range ({min_val} - {max_val})")
        # You can add custom explanations here for each variable, e.g.:
        if feature == 'Age':
            st.write("  - Description: Age of the patient in years.")
        elif feature == 'Blood Pressure':
            st.write("  - Description: The systolic blood pressure of the patient in mmHg.")
        elif feature == 'Cholesterol':
            st.write("  - Description: Cholesterol level of the patient in mg/dL.")
        # Add explanations for other variables in the same way

# Main function to run the app
def main():
    st.title("Hypertension Prediction with AI oleh Fabian J Manoppo")
    
    # Load and display the dataset
    data = load_data()
    
    if data is not None:
        st.write("### Dataset Overview")
        st.write(data.head())
        
        # Check if target column exists
        st.write("### Dataset Columns")
        st.write(data.columns)
        
        # Let the user select the target column
        target = st.selectbox("Select the target column (Hypertension indicator)", data.columns)

        # Let the user select the features to include
        features = st.multiselect("Select the feature columns", data.columns.difference([target]))

        if len(features) == 0:
            st.warning("Please select at least one feature.")
        else:
            # Handle missing values (if any) in the selected columns
            data = data[features + [target]].dropna()

            # Display information about each variable (range, description)
            display_variable_info(data, features)
            
            # Check for categorical columns and encode them
            for col in data[features].columns:
                if data[col].dtype == 'object':
                    st.write(f"Encoding categorical column: {col}")
                    data[col] = LabelEncoder().fit_transform(data[col])

            # Split data into features (X) and target (y)
            X = data[features]
            y = data[target]

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train a Random Forest classifier
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)

            # Make predictions and evaluate the model
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")

            # User input for making predictions
            st.write("### Enter Patient Data for Hypertension Prediction")
            user_input = []
            for feature in features:
                value = st.number_input(f"Enter {feature} (Range: {X[feature].min()} - {X[feature].max()})", 
                                        min_value=float(X[feature].min()), 
                                        max_value=float(X[feature].max()), 
                                        value=float(X[feature].mean()))
                user_input.append(value)
            
            # Make prediction based on user input
            if st.button("Predict"):
                input_data = pd.DataFrame([user_input], columns=features)
                prediction = model.predict(input_data)
                st.write(f"Prediction: {'Hypertensive' if prediction[0] == 1 else 'Non-Hypertensive'}")
    else:
        st.warning("Dataset could not be loaded. Please check the dataset file and try again.")

# Run the app
if __name__ == "__main__":
    main()
