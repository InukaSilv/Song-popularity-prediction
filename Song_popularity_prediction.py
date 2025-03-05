# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os

dataset = r"Song popularity\tracks.csv"
data = pd.read_csv(dataset)
print("First 5 rows of the dataset:")
print(data.head())

print("\nDataset shape:", data.shape)
print("\nMissing values:\n", data.isnull().sum())
print("\nBasic statistics:")
print(data.describe())

# Drop unnecessary columns 
data = data.drop(columns=['id', 'name', 'artists']) 
# Handle missing values 
data = data.dropna() 

# Select features (X) and target (y)
X = data[['danceability', 'energy', 'loudness', 'tempo']] 
y = data['popularity'] 

print("\nFeatures (X):")
print(X.head())
print("\nTarget (y):")
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

model = LinearRegression()
model.fit(X_train, y_train)

# Display the model's coefficients
print("\nModel coefficients:", model.coef_)

y_pred = model.predict(X_test)

print("\nPredictions:")
print(y_pred[:5])

mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error:", mse)

r2 = r2_score(y_test, y_pred)
print("R-squared:", r2)

plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")
plt.title("Actual vs. Predicted Popularity")
plt.show()