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
        
        # Add custom explanations for each variable
        if feature == 'ca':
            st.write("  - Full Name: Number of Major Vessels (0-4) Colored by Fluoroscopy")
            st.write("  - Description: Number of major blood vessels colored by fluoroscopy.")
        elif feature == 'cp':
            st.write("  - Full Name: Chest Pain Type")
            st.write("  - Description: 0: Typical Angina, 1: Atypical Angina, 2: Non-anginal Pain, 3: Asymptomatic")
        elif feature == 'fbs':
            st.write("  - Full Name: Fasting Blood Sugar")
            st.write("  - Description: 1 if fasting blood sugar > 120 mg/dL, 0 otherwise.")
        elif feature == 'restecg':
            st.write("  - Full Name: Resting Electrocardiographic Results")
            st.write("  - Description: 0: Normal, 1: ST-T abnormality, 2: Left ventricular hypertrophy.")
        elif feature == 'slope':
            st.write("  - Full Name: Slope of the Peak Exercise ST Segment")
            st.write("  - Description: 0: Upsloping, 1: Flat, 2: Downsloping.")
        elif feature == 'target':
            st.write("  - Full Name: Heart Disease Diagnosis Indicator")
            st.write("  - Description: 0: No heart disease, 1: Presence of heart disease.")
        elif feature == 'thalach':
            st.write("  - Full Name: Maximum Heart Rate Achieved")
            st.write("  - Description: The maximum heart rate achieved during the test.")
        elif feature == 'trestbps':
            st.write("  - Full Name: Resting Blood Pressure (in mm Hg)")
            st.write("  - Description: The resting blood pressure in mm Hg when admitted to the hospital.")
            
# Main function to run the app
def main():
    st.title("Hypertension Prediction with Machine Learning AI oleh Fabian J Manoppo")

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
