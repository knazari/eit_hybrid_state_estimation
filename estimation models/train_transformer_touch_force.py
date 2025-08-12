import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# === Load data ===
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
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(-1)  # (N, 208, 1)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(-1)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

# === Transformer Model ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class TransformerRegressor(nn.Module):
    def __init__(self, seq_len=208, d_model=64, nhead=8, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):  # x: (B, 208, 1)
        x = self.input_proj(x)           # (B, 208, d_model)
        x = self.pos_encoder(x)          # (B, 208, d_model)
        x = self.transformer(x)          # (B, 208, d_model)
        x = x.mean(dim=1)                # Global average pooling
        return self.regressor(x)         # (B, 3)

model = TransformerRegressor()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()

# === Training loop ===
epochs = 300
batch_size = 64
print("🔧 Training Transformer Regressor...")

for epoch in range(epochs):
    model.train()
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

print("\n📊 Transformer Model Performance:")
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
    plt.title(f"Transformer Prediction: {labels[i]}")
    plt.grid(True)
    plt.axis("equal")
    plt.plot([min(y_test_unscaled[:, i]), max(y_test_unscaled[:, i])],
             [min(y_test_unscaled[:, i]), max(y_test_unscaled[:, i])],
             'r--')
    plt.tight_layout()
    plt.savefig(f"data/transformer_pred_{labels[i]}.png", dpi=300)
    plt.close()

print("✅ Training complete. Plots saved in 'data/'")
