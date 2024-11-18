import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Define the number of samples
n_samples = 1000

# Generate physiological variables
ear_thickness = np.random.uniform(1, 5, n_samples)  # mm
ear_luminosity = np.random.uniform(0.5, 1.5, n_samples)  # arbitrary units
age = np.random.uniform(20, 80, n_samples)  # years
bmi = np.random.uniform(18, 35, n_samples)  # Body Mass Index (kg/m^2)
blood_pressure = np.random.uniform(90, 140, n_samples)  # Systolic BP (mmHg)

# Calulate measured impedance useing linear relationship + noise
measured_impedance = (
    1000 +
    (50 * ear_thickness) -
    (10 * ear_luminosity) +
    (2 * age) +
    (3 * bmi) +
    (0.5 * blood_pressure) +
    np.random.normal(0, 10, n_samples)  # Measurement noise
)