#!/usr/bin/env python3

# Predict California House Prices Using Linear Regression

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Load California housing data
data = fetch_california_housing()

# Convert to a pandas DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Target'] = data.target  # Target is the median house value

# Show the first 5 rows
print(df.head())

# Features and target
X = df.drop('Target', axis=1)
y = df['Target']

# Split data (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create the model
model = LinearRegression()

# Train it using the training data
model.fit(X_train, y_train)

