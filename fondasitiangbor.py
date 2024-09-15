import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv('datasetteori.csv')

# Define features (X) and targets (y)
X = df[['diameter', 'length', 'nspt1', 'nspt2', 'nspt3']]  # Input features
y = df[['qa', 'sa', 'qh', 'yh', 'bm']]  # Target variables

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create RandomForestRegressor model and wrap it in MultiOutputRegressor
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)

# Allow manual input for features
print("Please input the values for the following features:")
diameter = float(input("Diameter: "))
length = float(input("Length: "))
nspt1 = float(input("NSPT1: "))
nspt2 = float(input("NSPT2: "))
nspt3 = float(input("NSPT3: "))

# Create a NumPy array of the manually input features
input_data = np.array([[diameter, length, nspt1, nspt2, nspt3]])

# Predict the target values based on the input
predicted_values = model.predict(input_data)

# Display the predicted results
predicted_qa, predicted_sa, predicted_qh, predicted_yh, predicted_bm = predicted_values[0]
print("\nPredicted values based on your inputs:")
print(f"Predicted qa: {predicted_qa}")
print(f"Predicted sa: {predicted_sa}")
print(f"Predicted qh: {predicted_qh}")
print(f"Predicted yh: {predicted_yh}")
print(f"Predicted bm: {predicted_bm}")

# Optionally, evaluate the model on test data and show performance metrics
y_pred_test = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred_test, multioutput='raw_values')
mae = mean_absolute_error(y_test, y_pred_test, multioutput='raw_values')
r2 = r2_score(y_test, y_pred_test, multioutput='raw_values')

print("\nModel performance on test set:")
print(f"Mean Squared Error (MSE) for qa, sa, qh, yh, bm: {mse}")
print(f"Mean Absolute Error (MAE) for qa, sa, qh, yh, bm: {mae}")
print(f"R-squared (R²) for qa, sa, qh, yh, bm: {r2}")
