import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# ===========================
# Config: choose dataset
# ===========================
DATA_SOURCE = "real"   # "sim" or "real"

if DATA_SOURCE == "sim":
    DATA_PATH = "/home/kiyanoush/eit-experiments/data/touch_force_dataset.csv"
    OUT_DIR = "results_cnntr_sim"
elif DATA_SOURCE == "real":
    DATA_PATH = "data/merged_with_probe.csv"  # <-- set to your CSV (or merged CSV)
    OUT_DIR = "results_cnntr_real"
else:
    raise ValueError("DATA_SOURCE must be 'sim' or 'real'")

os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("✅ Using device:", device)

# ===========================
# Load dataset
# ===========================
df = pd.read_csv(DATA_PATH)

if DATA_SOURCE == "sim":
    X = df.iloc[:, :-3].values
    y = df.iloc[:, -3:].values
    label_names = ["x", "y", "force"]
else:
   # 1) Pick EIT columns (exclude 't_eit' if present)
    eit_cols = [c for c in df.columns if c.startswith("eit_") and c.lower() != "t_eit"]
    if len(eit_cols) == 0:
        raise RuntimeError("No EIT columns found (eit_0, eit_1, ...). Did you point DATA_PATH to a real CSV?")

    # 2) Remove EIT channels that are constant / near-constant
    eps = 1e-9
    variable_eit_cols = []
    for c in eit_cols:
        std_c = df[c].std(skipna=True)
        if df[c].nunique(dropna=False) > 1 and std_c is not None and std_c > eps:
            variable_eit_cols.append(c)

    if len(variable_eit_cols) == 0:
        raise RuntimeError("All EIT channels are constant/near-constant after filtering. Check the data.")

    # 3) Build targets
    targets = df[["contact_u_m", "contact_v_m", "force_n"]].copy()

    # Masks
    mask_contact   = targets["force_n"].notna() & (targets["force_n"] >= 0.5)     # keep for training
    mask_baseline  = targets["force_n"].notna() & (targets["force_n"] <  0.05)     # use to compute baseline

    # 4) Compute baseline vector v_ref over baseline rows (per-channel mean)
    if mask_baseline.sum() == 0:
        raise RuntimeError("No 'no-touch' rows (force_n < 0.5) found to compute baseline.")
    v_ref = df.loc[mask_baseline, variable_eit_cols].astype(np.float32).to_numpy()
    # nanmean guards against any stray NaNs
    v_ref = np.nanmean(v_ref, axis=0, dtype=np.float64).astype(np.float32)  # shape: [D]

    # 5) Apply row filter (contact cases), subtract baseline from EIT signals
    X_raw = df.loc[mask_contact, variable_eit_cols].astype(np.float32).to_numpy()  # [N, D]
    X = (X_raw - v_ref[None, :]).astype(np.float32)                                # baseline-subtracted features

    y = targets.loc[mask_contact, :].to_numpy(dtype=np.float32)
    label_names = ["x", "y", "force"]

    # (Optional) Keep v_ref for later use (e.g., saving with model to normalize test-time inputs)
    baseline_vector = v_ref  # np.ndarray shape [D]
    
    # (Optional) quick diagnostics
    print(f"EIT channels total: {len(eit_cols)} | kept: {len(variable_eit_cols)} | removed: {len(eit_cols) - len(variable_eit_cols)}")
    print(f"Samples total: {len(df)} | kept: {mask_contact.sum()} | dropped: {(~mask_contact).sum()}")

n_features = X.shape[1]
print(f"Samples: {X.shape[0]} | EIT length: {n_features}")

# Keep a copy of RAW (unscaled) EIT for the physics loss
X_raw = X.copy()

# ===========================
# Train/test split & scaling
# ===========================
X_train, X_test, y_train, y_test, X_train_raw, X_test_raw = train_test_split(
    X, y, X_raw, test_size=0.2, random_state=42
)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled  = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled  = scaler_y.transform(y_test)

# ===========================
# Tensors
# ===========================
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)[:, None, :].to(device)  # (N,1,L)
X_test_tensor  = torch.tensor(X_test_scaled,  dtype=torch.float32)[:, None, :].to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
y_test_tensor  = torch.tensor(y_test_scaled,  dtype=torch.float32).to(device)

# RAW EIT (unscaled) for physics loss
X_train_raw_tensor = torch.tensor(X_train_raw, dtype=torch.float32).to(device)             # (N,L)

# ===========================
# CNN model (length-adaptive)
# ===========================
class CNNRegressor(nn.Module):
    def __init__(self, input_len, out_dim=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),   # L -> L/2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),   # L -> L/4
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_len)
            flat_dim = self.features(dummy).view(1, -1).shape[1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )
    def forward(self, x):  # x: (B,1,L)
        x = self.features(x)
        return self.head(x)

model = CNNRegressor(input_len=n_features, out_dim=y_train_tensor.shape[1]).to(device)

# ===========================
# Train
# ===========================
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_pred = nn.MSELoss()
lambda_phys = 0.01           # weight of physics-informed loss
epochs = 700
batch_size = 64

print("🔧 Training Physics-Informed CNN (monotonicity loss)...")
model.train()
N = X_train_tensor.size(0)

for epoch in range(epochs):
    perm = torch.randperm(N, device=device)
    for i in range(0, N, batch_size):
        idx = perm[i:i+batch_size]
        xb = X_train_tensor[idx]          # (B,1,L) scaled
        yb = y_train_tensor[idx]          # (B,3)   scaled
        xb_raw = X_train_raw_tensor[idx]  # (B,L)   unscaled

        optimizer.zero_grad()
        y_pred = model(xb)                 # scaled predictions
        loss_p = loss_pred(y_pred, yb)

        # ---- Physics-informed monotonicity term ----
        # Compare norm of RAW ΔV vector per sample with predicted force (both standardized within batch)
        v_norm = torch.norm(xb_raw, dim=1)             # (B,)
        f_pred = y_pred[:, 2]                          # (B,) scaled force
        # standardize within the batch (avoid zero-divide with eps)
        eps = 1e-8
        v_norm_std = (v_norm - v_norm.mean()) / (v_norm.std() + eps)
        f_pred_std = (f_pred - f_pred.mean()) / (f_pred.std() + eps)
        loss_phys = nn.MSELoss()(v_norm_std, f_pred_std)

        total = loss_p + lambda_phys * loss_phys
        total.backward()
        optimizer.step()

    if epoch % 50 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:3d} | Pred: {loss_p.item():.6f} | Phys: {loss_phys.item():.6f}")

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

print("\n📊 Physics-Informed CNN Performance:")
for i, lbl in enumerate(label_names):
    print(f"MAE - {lbl:5s}: {mae[i]:.4f}")
for i, lbl in enumerate(label_names):
    print(f"R²  - {lbl:5s}: {r2[i]:.4f}")

# ===========================
# Plots
# ===========================
for i, lbl in enumerate(label_names):
    plt.figure(figsize=(4.5, 4))
    plt.scatter(y_test_unscaled[:, i], y_pred[:, i], alpha=0.7)
    plt.xlabel(f"True {lbl}")
    plt.ylabel(f"Predicted {lbl}")
    plt.title(f"Phys-CNN Prediction: {lbl}")
    plt.grid(True)
    plt.axis("equal")
    lo, hi = float(np.min(y_test_unscaled[:, i])), float(np.max(y_test_unscaled[:, i]))
    plt.plot([lo, hi], [lo, hi], "r--")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"physcnn_pred_{lbl}.png"), dpi=300)
    plt.close()

print(f"✅ Training complete. Outputs saved in '{OUT_DIR}/'")
