import pandas as pd
import numpy as np

# Define number of samples
num_samples = 25

# Generate synthetic data
np.random.seed(42)
Feature1 = np.random.uniform(4, 8, num_samples)  # Random values between 4 and 8
Feature2 = np.random.uniform(2, 4, num_samples)  # Random values between 2 and 4
Target = 2.5 * Feature1 + 1.8 * Feature2 + np.random.normal(0, 0.5, num_samples)  # Linear relation with noise

# Create DataFrame
df = pd.DataFrame({"Feature1": Feature1, "Feature2": Feature2, "Target": Target})

# Save to CSV
df.to_csv("multiple_linear_regression_data.csv", index=False)

print("CSV file saved as 'multiple_linear_regression_data.csv'")
