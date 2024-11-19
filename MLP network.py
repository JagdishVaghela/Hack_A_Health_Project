import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

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

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the neural network for bias prediction
model = Sequential([
    Dense(256, input_dim=X_train_scaled.shape[1], activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1)  # Output layer for bias prediction
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Add early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
