import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from model import LSTMModel

# Load data
X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")

X_test = np.load("data/X_test.npy")
y_test = np.load("data/y_test.npy")

# Convert to tensors
X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_test = torch.tensor(
    y_test,
    dtype=torch.float32
)

# DataLoader
batch_size = 256

train_dataset = TensorDataset(
    X_train,
    y_train
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

# Model
input_size = X_train.shape[2]

model = LSTMModel(input_size)

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 20

train_losses = []

# Training loop
for epoch in range(epochs):

    model.train()

    epoch_loss = 0

    for batch_X, batch_y in train_loader:

        outputs = model(batch_X)

        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)

    train_losses.append(avg_loss)

    print(f"Epoch {epoch+1}/{epochs}")
    print(f"Loss: {avg_loss:.6f}")

# Save model
torch.save(
    model.state_dict(),
    "lstm_model.pth"
)

# Evaluate on test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)

print(f"Test Loss: {test_loss.item():.6f}")

# Plot training loss
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.tight_layout()
plt.savefig("results/train_loss.png")
plt.show()

print("Training completed.")