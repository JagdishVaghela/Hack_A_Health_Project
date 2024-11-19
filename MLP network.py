import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

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

# Train the model
history = model.fit(
    X_train_scaled, y_train, 
    validation_data=(X_test_scaled, y_test), 
    epochs=100, 
    batch_size=32, 
    verbose=1, 
    callbacks=[early_stopping]
)

# Evaluate the model on the test set
loss, mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Mean Absolute Error for Bias Prediction: {mae}")

# Predict bias for the test set
predicted_bias = model.predict(X_test_scaled).flatten()

# Adjust impedance using predicted bias
adjusted_impedance = Z_test + predicted_bias

# Train a regression model for cholesterol prediction (optional improvement)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(np.array(adjusted_impedance).reshape(-1, 1), actual_cholesterol_test)

# Predict cholesterol using the regression model
predicted_cholesterol = regressor.predict(np.array(adjusted_impedance).reshape(-1, 1))

# Calculate Mean Absolute Error for cholesterol prediction
mae_cholesterol = np.mean(np.abs(actual_cholesterol_test - predicted_cholesterol))
print(f"Mean Absolute Error in Cholesterol Prediction: {mae_cholesterol}")

# Plot training and validation loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs', fontsize=14)
plt.ylabel('Loss (MSE)', fontsize=14)
plt.title('Training and Validation Loss', fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.5)
plt.show()

# Plot actual vs predicted cholesterol
plt.figure(figsize=(10, 6))
plt.scatter(actual_cholesterol_test, predicted_cholesterol, color='blue', alpha=0.6, label='Predicted Cholesterol')
plt.plot([actual_cholesterol_test.min(), actual_cholesterol_test.max()],
         [actual_cholesterol_test.min(), actual_cholesterol_test.max()],
         color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel('Actual Cholesterol', fontsize=14)
plt.ylabel('Predicted Cholesterol', fontsize=14)
plt.title('Actual vs. Predicted Cholesterol', fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.5)
plt.show()
