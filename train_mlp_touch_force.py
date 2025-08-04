import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# === Step 1: Load and split dataset ===
df = pd.read_csv("data/touch_force_dataset.csv")
X = df.iloc[:, :-3].values
y = df.iloc[:, -3:].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Step 2: Standardize ===
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# === Step 3: Convert to PyTorch tensors ===
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# === Step 4: Define the MLP model ===
class MLP(nn.Module):
    def __init__(self, input_size=208, output_size=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        return self.net(x)

model = MLP()

# === Step 5: Train the model ===
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 500
batch_size = 64

for epoch in range(epochs):
    permutation = torch.randperm(X_train_tensor.size(0))
    for i in range(0, X_train_tensor.size(0), batch_size):
        indices = permutation[i:i + batch_size]
        x_batch = X_train_tensor[indices]
        y_batch = y_train_tensor[indices]

        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

    if epoch % 50 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")

# === Step 6: Evaluate ===
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor).numpy()

# Inverse scale
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_unscaled = scaler_y.inverse_transform(y_test_tensor.numpy())

# === Step 7: Metrics ===
mae = mean_absolute_error(y_test_unscaled, y_pred, multioutput="raw_values")
r2 = r2_score(y_test_unscaled, y_pred, multioutput="raw_values")

print("\n📊 MLP Performance (on Test Set):")
print(f"MAE - X:     {mae[0]:.4f}")
print(f"MAE - Y:     {mae[1]:.4f}")
print(f"MAE - Force: {mae[2]:.4f}")
print(f"R²  - X:     {r2[0]:.4f}")
print(f"R²  - Y:     {r2[1]:.4f}")
print(f"R²  - Force: {r2[2]:.4f}")

# === Step 8: Plot predictions ===
labels = ["x", "y", "force"]
for i in range(3):
    plt.figure(figsize=(4.5, 4))
    plt.scatter(y_test_unscaled[:, i], y_pred[:, i], alpha=0.7)
    plt.xlabel(f"True {labels[i]}")
    plt.ylabel(f"Predicted {labels[i]}")
    plt.title(f"MLP Prediction: {labels[i]}")
    plt.grid(True)
    plt.axis("equal")
    plt.plot([min(y_test_unscaled[:, i]), max(y_test_unscaled[:, i])],
             [min(y_test_unscaled[:, i]), max(y_test_unscaled[:, i])],
             'r--')  # y=x line
    plt.tight_layout()
    plt.savefig(f"data/mlp_prediction_{labels[i]}.png", dpi=300)
    plt.close()

print("✅ MLP training and evaluation complete. Plots saved to data/")

# Extract true x, y, and force prediction error
x_coords = y_test_unscaled[:, 0]
y_coords = y_test_unscaled[:, 1]
force_true = y_test_unscaled[:, 2]
force_pred = y_pred[:, 2]
force_error = np.abs(force_pred - force_true)

# Plot error vs location
plt.figure(figsize=(6, 6))
sc = plt.scatter(x_coords, y_coords, c=force_error, cmap='hot', s=60, edgecolors='k')
plt.colorbar(sc, label="Force prediction error (N)")
plt.xlabel("Touch X")
plt.ylabel("Touch Y")
plt.title("MLP Force Prediction Error by Touch Location")
plt.gca().set_aspect("equal")
plt.grid(True)
plt.savefig("data/force_error_map.png", dpi=300)
plt.close()
print("📍 Saved spatial force error heatmap as 'data/force_error_map.png'")

