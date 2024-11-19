import pandas as pd

# Load the dataset
data = pd.read_csv("Enhanced_Impedance_Dataset.csv")

# Define input features, target (bias), and other relevant columns
X = data[["Ear Thickness", "Ear Luminosity", "Age", "BMI", "Blood Pressure"]]
y = data["Bias"]  # Target for bias prediction
Z = data["Measured Impedance"]  # Original impedance values
actual_cholesterol = data["Measured Cholesterol"]  # Actual cholesterol levels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
Z_train, Z_test = train_test_split(Z, test_size=0.2, random_state=42)
actual_cholesterol_train, actual_cholesterol_test = train_test_split(
    actual_cholesterol, test_size=0.2, random_state=42
)

