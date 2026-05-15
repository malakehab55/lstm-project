import os
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import MinMaxScaler

# Create folders if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Read dataset
df = pd.read_csv("data/network_traffic.csv")

# Sort by Time to ensure correct time series order
df = df.sort_values('Time').reset_index(drop=True)

# Use only Length feature
df = df[['Length']]

# Scaling
scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(df)

# Save scaler for inverse_transform later
joblib.dump(scaler, "data/scaler.pkl")
print("Scaler saved.")

# Sequence creation
sequence_length = 40

X = []
y = []

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i])

X = np.array(X)
y = np.array(y)

# Train/Test split
split = int(0.8 * len(X))

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# Save arrays
np.save("data/X_train.npy", X_train)
np.save("data/X_test.npy", X_test)

np.save("data/y_train.npy", y_train)
np.save("data/y_test.npy", y_test)

print("Preprocessing completed.")

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)