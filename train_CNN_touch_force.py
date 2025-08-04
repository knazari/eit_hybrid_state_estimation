import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# === Load dataset ===
df = pd.read_csv("data/touch_force_dataset.csv")
X = df.iloc[:, :-3].values
y = df.iloc[:, -3:].values

# === Split and scale ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# === Convert to tensors ===
X_train_tensor = torch.tensor(X_train_scaled[:, np.newaxis, :], dtype=torch.float32)  # shape: [N, 1, 208]
X_test_tensor = torch.tensor(X_test_scaled[:, np.newaxis, :], dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# === CNN model ===
class CNN1DRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Flatten(),
            nn.Linear(64 * 52, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.net(x)

model = CNN1DRegressor()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# === Training ===
epochs = 300
batch_size = 64
print("\n🔧 Training 1D CNN...")
for epoch in range(epochs):
    perm = torch.randperm(X_train_tensor.size(0))
    for i in range(0, X_train_tensor.size(0), batch_size):
        idx = perm[i:i+batch_size]
        x_batch = X_train_tensor[idx]
        y_batch = y_train_tensor[idx]

        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()

    if epoch % 50 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")

# === Evaluation ===
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor).numpy()

y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_unscaled = scaler_y.inverse_transform(y_test_tensor.numpy())

# === Metrics ===
mae = mean_absolute_error(y_test_unscaled, y_pred, multioutput="raw_values")
r2 = r2_score(y_test_unscaled, y_pred, multioutput="raw_values")

print("\n📊 CNN1D Performance:")
print(f"MAE - X:     {mae[0]:.4f}")
print(f"MAE - Y:     {mae[1]:.4f}")
print(f"MAE - Force: {mae[2]:.4f}")
print(f"R²  - X:     {r2[0]:.4f}")
print(f"R²  - Y:     {r2[1]:.4f}")
print(f"R²  - Force: {r2[2]:.4f}")

# === Plot predictions ===
labels = ["x", "y", "force"]
for i in range(3):
    plt.figure(figsize=(4.5, 4))
    plt.scatter(y_test_unscaled[:, i], y_pred[:, i], alpha=0.7)
    plt.xlabel(f"True {labels[i]}")
    plt.ylabel(f"Predicted {labels[i]}")
    plt.title(f"CNN1D Prediction: {labels[i]}")
    plt.grid(True)
    plt.axis("equal")
    plt.plot([min(y_test_unscaled[:, i]), max(y_test_unscaled[:, i])],
             [min(y_test_unscaled[:, i]), max(y_test_unscaled[:, i])],
             'r--')  # y = x line
    plt.tight_layout()
    plt.savefig(f"data/cnn1d_prediction_{labels[i]}.png", dpi=300)
    plt.close()

print("✅ Training complete. Plots saved in 'data/'")
