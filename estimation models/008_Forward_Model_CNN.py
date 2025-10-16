# physics_aware_cnn_unified.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------
# PyEIT
# ---------------------------
from pyeit.mesh import create
from pyeit.eit.fem import EITForward
from pyeit.mesh.wrapper import set_perm, PyEITAnomaly_Circle
from pyeit.eit.protocol import create as create_protocol

# ===========================
# Config
# ===========================
DATA_SOURCE = "real"   # "sim" or "real"

if DATA_SOURCE == "sim":
    DATA_PATH = "data/touch_force_dataset.csv"
    OUT_DIR = "outputs_phys_sim"
else:
    DATA_PATH = "data/merged_with_probe.csv"  # real CSV (can be merged)
    OUT_DIR = "outputs_phys_real"

os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# If training on real data, your labels (x,y) are in meters. Optionally scale them to mesh units.
# With a 9.4 cm radius sensor, a reasonable mapping is: mesh unit = meters / 0.094
REAL_POS_SCALE = 1.0 / 0.094  # set to 1.0 to disable

# Physics loss weight
LAMBDA_FWD = 0.5

# --- Probe radii (meters) for categorical column 'probe_type' ---
# EDIT to match your three probes:
PROBE_RADIUS_M = {
    "small": 0.007,   # 6.5 mm
    "medium": 0.0100,  # 10  mm
    "large": 0.0150,   # 13  mm
}
# If you prefer a smaller effective contact than the physical tip:
CONTACT_RADIUS_SCALE = 1.0  # e.g., 0.8

# ===========================
# Load data
# ===========================
df = pd.read_csv(DATA_PATH)

if DATA_SOURCE == "sim":
    # Features = voltages; last 3 labels = x,y,force
    X = df.iloc[:, :-3].values
    y = df.iloc[:, -3:].values
    label_names = ["x", "y", "force"]
    # Provide a constant radius for simulated data (edit if your sim has it)
    probe_r_all = np.full(len(X), 0.0100, dtype=float)  # 10 mm default
else:
    # Real: features = all EIT channels; targets = contact_u_m, contact_v_m, force_n
    eit_cols = [c for c in df.columns if c.startswith("eit_")]
    if len(eit_cols) == 0:
        raise RuntimeError("No EIT columns (eit_*) found in real dataset.")
    y_df = df[["contact_u_m", "contact_v_m", "force_n"]].copy()
    keep = ~y_df["force_n"].isna()
    X = df.loc[keep, eit_cols].values
    y = y_df.loc[keep, ["contact_u_m", "contact_v_m", "force_n"]].values
    label_names = ["x", "y", "force"]

    # ---- derive per-sample radius (meters) ----
    if "probe_radius_m" in df.columns:
        probe_r_all = df.loc[keep, "probe_radius_m"].astype(float).values
    elif "probe_type" in df.columns:
        valid_types = set(df.loc[keep, "probe_type"].dropna().unique())
        if not valid_types.issubset(PROBE_RADIUS_M.keys()):
            raise ValueError(f"Found probe types {valid_types} not in PROBE_RADIUS_M mapping.")
        probe_r_all = df.loc[keep, "probe_type"].map(PROBE_RADIUS_M).astype(float).values
    else:
        raise ValueError("Need either 'probe_radius_m' or 'probe_type' column for real dataset.")

n_features = X.shape[1]
print(f"Dataset: {DATA_SOURCE} | Samples: {X.shape[0]} | EIT length: {n_features}")

# Keep a copy of the **raw** voltages for forward-model loss
X_raw = X.copy()

# ===========================
# Split & scale
# ===========================
X_train, X_test, y_train, y_test, X_train_raw, X_test_raw, r_train, r_test = train_test_split(
    X, y, X_raw, probe_r_all, test_size=0.2, random_state=42
)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled  = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled  = scaler_y.transform(y_test)

# ===========================
# Tensors for CNN ([N, 1, L])
# ===========================
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)[:, None, :].to(device)
X_test_tensor  = torch.tensor(X_test_scaled,  dtype=torch.float32)[:, None, :].to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
y_test_tensor  = torch.tensor(y_test_scaled,  dtype=torch.float32).to(device)

# Raw voltages + radii
X_train_raw_tensor = torch.tensor(X_train_raw, dtype=torch.float32).to(device)
probe_r_train_tensor = torch.tensor(r_train, dtype=torch.float32).to(device)
probe_r_test_tensor  = torch.tensor(r_test,  dtype=torch.float32).to(device)

# ===========================
# CNN (length-adaptive head)
# ===========================
class CNNRegressor(nn.Module):
    def __init__(self, input_len, out_dim=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),   # L -> L/2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),   # L -> L/4
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_len)
            flat_dim = self.conv(dummy).view(1, -1).shape[1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.head(x)

model = CNNRegressor(input_len=n_features, out_dim=y_train_tensor.shape[1]).to(device)

# ===========================
# PyEIT forward model
# ===========================
mesh_obj = create(n_el=16, h0=0.1)
protocol = create_protocol(n_el=16, dist_exc=1, step_meas=1)
fwd = EITForward(mesh_obj, protocol)

def simulate_voltage_from_prediction(x_phys, y_phys, force_phys, r_contact_m):
    """
    Simulate voltages from predicted (x,y,force) with contact radius r_contact_m (meters).
    For real data, (x,y,r) are scaled to mesh units via REAL_POS_SCALE (and CONTACT_RADIUS_SCALE).
    """
    if DATA_SOURCE == "real":
        x_sim = float(x_phys) * REAL_POS_SCALE
        y_sim = float(y_phys) * REAL_POS_SCALE
        r_sim = float(r_contact_m) * CONTACT_RADIUS_SCALE * REAL_POS_SCALE
    else:
        x_sim = float(x_phys)
        y_sim = float(y_phys)
        r_sim = float(r_contact_m) * CONTACT_RADIUS_SCALE

    # Simple mapping: perm = 1 + k*force (tune if you have calibration)
    anomaly = PyEITAnomaly_Circle(center=[x_sim, y_sim], r=r_sim, perm=1.0 + float(force_phys))
    mesh_new = set_perm(mesh_obj, anomaly=[anomaly])
    v_sim = fwd.solve_eit(perm=mesh_new.perm)
    return torch.tensor(v_sim, dtype=torch.float32, device=device)

# Directory to save the best model
os.makedirs("models", exist_ok=True)
best_model_path = "models/FM_CNN_model.pth"

# ===========================
# Train
# ===========================
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_pred = nn.MSELoss()
loss_fwd  = nn.MSELoss()

epochs = 500
batch_size = 64
best_loss = float("inf")

print("Training with physics forward consistency loss...")
model.train()
N = X_train_tensor.size(0)

for epoch in range(epochs):
    perm = torch.randperm(N, device=device)
    total_loss = 0.0

    for i in range(0, N, batch_size):
        idx = perm[i:i+batch_size]
        xb = X_train_tensor[idx]          # [B,1,L] (scaled)
        yb = y_train_tensor[idx]          # [B,3]    (scaled)
        xb_raw = X_train_raw_tensor[idx]  # [B,L]    (unscaled)
        r_b = probe_r_train_tensor[idx]   # [B]      (meters)

        optimizer.zero_grad()
        yb_pred = model(xb)               # scaled predictions
        pred_mse = loss_pred(yb_pred, yb)

        # Physics loss (per-sample)
        phys_loss = 0.0
        with torch.no_grad():
            yb_pred_phys = torch.tensor(
                scaler_y.inverse_transform(yb_pred.detach().cpu().numpy()),
                dtype=torch.float32,
                device=device,
            )

        B = yb_pred_phys.size(0)
        for j in range(B):
            x_phys, y_phys, f_phys = yb_pred_phys[j].tolist()
            r_contact_m = float(r_b[j].item())
            v_sim = simulate_voltage_from_prediction(x_phys, y_phys, f_phys, r_contact_m)
            v_meas = xb_raw[j]
            if v_sim.numel() != v_meas.numel():
                L = min(v_sim.numel(), v_meas.numel())
                phys_loss += loss_fwd(v_sim[:L], v_meas[:L])
            else:
                phys_loss += loss_fwd(v_sim, v_meas)

        phys_loss = phys_loss / max(1, B)
        total = pred_mse + LAMBDA_FWD * phys_loss
        total.backward()
        optimizer.step()

        total_loss += float(total.item()) * yb.size(0)

    # ---- end of epoch: compute avg loss & save if improved ----
    epoch_loss = total_loss / N
    if epoch % 20 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:3d} | total loss: {epoch_loss:.6f}")

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_loss": best_loss,
            },
            best_model_path,
        )
        print(f"  ↳ ✅ new best ({best_loss:.6f}); saved to {best_model_path}")

# ===========================
# Evaluate
# ===========================
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor).cpu().numpy()

y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_unscaled = scaler_y.inverse_transform(y_test_tensor.cpu().numpy())

mae = mean_absolute_error(y_test_unscaled, y_pred, multioutput="raw_values")
r2  = r2_score(y_test_unscaled, y_pred, multioutput="raw_values")

print("\nFinal Model Performance:")
for i, name in enumerate(["x", "y", "force"]):
    print(f"MAE - {name:5s}: {mae[i]:.4f} | R² - {name:5s}: {r2[i]:.4f}")

# ===========================
# Plots
# ===========================
for i, name in enumerate(["x", "y", "force"]):
    plt.figure(figsize=(4.5, 4))
    plt.scatter(y_test_unscaled[:, i], y_pred[:, i], alpha=0.7)
    plt.xlabel(f"True {name}")
    plt.ylabel(f"Predicted {name}")
    plt.title(f"Physics-aware CNN: {name}")
    plt.grid(True)
    plt.axis("equal")
    lo, hi = float(np.min(y_test_unscaled[:, i])), float(np.max(y_test_unscaled[:, i]))
    plt.plot([lo, hi], [lo, hi], "r--")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"phys_cnn_pred_{name}.png"), dpi=300)
    plt.close()

print(f"✅ Done. Plots & outputs in '{OUT_DIR}/'.")
