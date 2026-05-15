import numpy as np
import torch
import matplotlib.pyplot as plt
import joblib

from model import LSTMModel

# Load test data
X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
)

# Load model
input_size = X_test.shape[2]

model = LSTMModel(input_size)

model.load_state_dict(
    torch.load("lstm_model.pth")
)

model.eval()

# Prediction
with torch.no_grad():

    predictions = model(X_test_tensor)

predictions = predictions.numpy()

# Inverse transform to original scale
scaler = joblib.load("data/scaler.pkl")
predictions_real = scaler.inverse_transform(predictions)
y_test_real = scaler.inverse_transform(y_test)

# Save predictions (scaled)
np.save(
    "predictions.npy",
    predictions
)

# Plot
plt.figure(figsize=(12,6))

plt.plot(
    y_test_real[:200],
    label="Actual Length"
)

plt.plot(
    predictions_real[:200],
    label="Predicted Length"
)

plt.legend()

plt.title(
    "Actual vs Predicted Network Traffic"
)

plt.savefig(
    "results/prediction_graph.png"
)

plt.show()