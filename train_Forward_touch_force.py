# physics_fwd_loss_cnn.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import os

# PyEIT specific imports
from pyeit.mesh import create
from pyeit.eit.interp2d import sim2pts
from pyeit.eit.fem import EITForward
from pyeit.mesh.wrapper import set_perm
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
from pyeit.eit.protocol import create as create_protocol

# === Paths and device ===
os.makedirs("outputs", exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# === Load data ===
df = pd.read_csv("data/touch_force_dataset.csv")
X = df.iloc[:, :-3].values
y = df.iloc[:, -3:].values

# === Preprocessing ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

X_train_tensor = torch.tensor(X_train_scaled[:, np.newaxis, :], dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled[:, np.newaxis, :], dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)

# === CNN Model ===
class CNNRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten(),
            nn.Linear(64 * 52, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
    def forward(self, x):
        return self.model(x)

# === PyEIT Setup ===
mesh_obj = create(n_el=16, h0=0.1)
el_pos = mesh_obj.el_pos
ex_mat = np.array([[i, (i+1)%16] for i in range(16)])
protocol = create_protocol(n_el=16, dist_exc=1, step_meas=1)
fwd = EITForward(mesh_obj, protocol)

# === Simulate voltage from prediction ===
def simulate_voltage_from_prediction(x, y, force):
    anomaly = PyEITAnomaly_Circle(center=[x, y], r=0.1, perm=1 + force)
    mesh_new = set_perm(mesh_obj, anomaly=[anomaly])
    v_sim = fwd.solve_eit(perm=mesh_new.perm)
    return torch.tensor(v_sim, dtype=torch.float32)

# === Training ===
model = CNNRegressor().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()
lambda_fwd = 0.01

print("Training with forward model consistency loss...")

for epoch in range(200):
    model.train()
    total_loss = 0
    for i in range(X_train_tensor.size(0)):
        x_sample = X_train_tensor[i:i+1]  # shape (1, 1, 208)
        y_true = y_train_tensor[i:i+1]
        v_measured = X_train_tensor[i].squeeze(0).detach().cpu().numpy()

        optimizer.zero_grad()
        y_pred = model(x_sample)
        pred_loss = loss_fn(y_pred, y_true)

        # === Forward simulation loss ===
        x_pred, y_pred_pos, force_pred = scaler_y.inverse_transform(y_pred.detach().cpu().numpy())[0]
        v_sim = simulate_voltage_from_prediction(x_pred, y_pred_pos, force_pred)
        v_meas = torch.tensor(X_train[i], dtype=torch.float32)  # unnormalized voltage
        fwd_loss = loss_fn(v_sim, v_meas)

        total = pred_loss + lambda_fwd * fwd_loss
        total.backward()
        optimizer.step()

        total_loss += total.item()

    if epoch % 20 == 0 or epoch == 199:
        print(f"Epoch {epoch:3d} | Total Loss: {total_loss:.4f}")

# === Evaluation ===
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor).cpu().numpy()

y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_unscaled = scaler_y.inverse_transform(y_test_tensor.cpu().numpy())

mae = mean_absolute_error(y_test_unscaled, y_pred, multioutput="raw_values")
r2 = r2_score(y_test_unscaled, y_pred, multioutput="raw_values")

print("\nFinal Model Performance:")
print(f"MAE - X: {mae[0]:.4f}, Y: {mae[1]:.4f}, Force: {mae[2]:.4f}")
print(f"R^2 - X: {r2[0]:.4f}, Y: {r2[1]:.4f}, Force: {r2[2]:.4f}")