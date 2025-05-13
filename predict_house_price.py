#!/usr/bin/env python3

# Predict California House Prices Using Linear Regression

import pandas as pd
from sklearn.datasets import fetch_california_housing

# Load California housing data
data = fetch_california_housing()

# Convert to a pandas DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target  # Target is the median house value

# Show the first 5 rows
print(df.head())
