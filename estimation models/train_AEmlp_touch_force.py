import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# === Load Data ===
df = pd.read_csv("data/touch_force_dataset.csv")
X = df.iloc[:, :-3].values
y = df.iloc[:, -3:].values

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Normalize ===
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# === Convert to Tensors ===
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# === Autoencoder ===
class Autoencoder(nn.Module):
    def __init__(self, input_dim=208, latent_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

autoencoder = Autoencoder()
ae_optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
ae_loss_fn = nn.MSELoss()

# === Train Autoencoder ===
print("\n🔧 Training Autoencoder...")
epochs = 1000
for epoch in range(epochs):
    autoencoder.train()
    x_recon = autoencoder(X_train_tensor)
    loss = ae_loss_fn(x_recon, X_train_tensor)
    ae_optimizer.zero_grad()
    loss.backward()
    ae_optimizer.step()
    if epoch % 50 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:3d} | AE Loss: {loss.item():.6f}")

# === Freeze Encoder and Extract Latent Features ===
autoencoder.eval()
with torch.no_grad():
    Z_train = autoencoder.encoder(X_train_tensor)
    Z_test = autoencoder.encoder(X_test_tensor)

# === Regressor on Latent Codes ===
class Regressor(nn.Module):
    def __init__(self, latent_dim=16, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )

    def forward(self, z):
        return self.net(z)

regressor = Regressor()
reg_optimizer = optim.Adam(regressor.parameters(), lr=1e-3)
reg_loss_fn = nn.MSELoss()

# === Train Regressor ===
print("\n🔧 Training Regressor on Latent Codes...")
epochs = 1000
for epoch in range(epochs):
    regressor.train()
    pred = regressor(Z_train)
    loss = reg_loss_fn(pred, y_train_tensor)
    reg_optimizer.zero_grad()
    loss.backward()
    reg_optimizer.step()
    if epoch % 50 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:3d} | Regressor Loss: {loss.item():.6f}")

# === Evaluate ===
regressor.eval()
with torch.no_grad():
    y_pred_scaled = regressor(Z_test).numpy()

y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_unscaled = scaler_y.inverse_transform(y_test_tensor.numpy())

# === Metrics ===
mae = mean_absolute_error(y_test_unscaled, y_pred, multioutput="raw_values")
r2 = r2_score(y_test_unscaled, y_pred, multioutput="raw_values")

print("\n📊 Autoencoder+Regressor Performance:")
print(f"MAE - X:     {mae[0]:.4f}")
print(f"MAE - Y:     {mae[1]:.4f}")
print(f"MAE - Force: {mae[2]:.4f}")
print(f"R²  - X:     {r2[0]:.4f}")
print(f"R²  - Y:     {r2[1]:.4f}")
print(f"R²  - Force: {r2[2]:.4f}")

# === Plot Predictions ===
labels = ["x", "y", "force"]
for i in range(3):
    plt.figure(figsize=(4.5, 4))
    plt.scatter(y_test_unscaled[:, i], y_pred[:, i], alpha=0.7)
    plt.xlabel(f"True {labels[i]}")
    plt.ylabel(f"Predicted {labels[i]}")
    plt.title(f"Autoencoder MLP: {labels[i]}")
    plt.grid(True)
    plt.axis("equal")
    plt.plot([min(y_test_unscaled[:, i]), max(y_test_unscaled[:, i])],
             [min(y_test_unscaled[:, i]), max(y_test_unscaled[:, i])],
             'r--')  # y=x
    plt.tight_layout()
    plt.savefig(f"data/ae_mlp_prediction_{labels[i]}.png", dpi=300)
    plt.close()

print("✅ Training complete. Prediction plots saved in 'data/'")
