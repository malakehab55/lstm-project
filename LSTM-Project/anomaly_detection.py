import numpy as np
import matplotlib.pyplot as plt
import joblib

# Load predictions
predictions = np.load("predictions.npy")

# Load actual values
y_test = np.load("data/y_test.npy")

# Inverse transform to original scale
scaler = joblib.load("data/scaler.pkl")
predictions_real = scaler.inverse_transform(predictions)
y_test_real = scaler.inverse_transform(y_test)

# Calculate error on real scale
error = np.abs(y_test_real - predictions_real)

# Threshold
threshold = np.mean(error) + 2 * np.std(error)

# Detect anomalies
anomalies = error > threshold

print("Total anomalies detected:")
print(np.sum(anomalies))

# Plot
plt.figure(figsize=(14,6))

plt.plot(
    error[:500],
    label="Prediction Error"
)

plt.axhline(
    threshold,
    color='r',
    linestyle='--',
    label="Threshold"
)

plt.legend()

plt.title("Anomaly Detection")

plt.savefig(
    "results/anomaly_graph.png"
)

plt.show()