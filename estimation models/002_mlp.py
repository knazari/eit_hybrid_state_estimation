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
DATA_SOURCE = "sim"   # "sim" or "real"

if DATA_SOURCE == "sim":
    DATA_PATH = "/home/kiyanoush/Projects/eit_hybrid_state_estimation/data/touch_force_dataset_sim2real.csv"
    OUT_DIR = "results_mlp_sim"
elif DATA_SOURCE == "real":
    # Point this to one of your collected CSVs (or a concatenated CSV of multiple trials)
    DATA_PATH = "data/merged_with_probe.csv"
    OUT_DIR = "results_mlp_real"
else:
    raise ValueError("DATA_SOURCE must be 'sim' or 'real'")

os.makedirs(OUT_DIR, exist_ok=True)

# ===========================
# Step 1: Load and split
# ===========================
df = pd.read_csv(DATA_PATH)

if DATA_SOURCE == "sim":
    # Simulated: last 3 columns = x, y, force
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


print(f"Dataset: {DATA_SOURCE} | Samples: {X.shape[0]} | Features: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===========================
# Step 2: Standardize
# ===========================
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# ===========================
# Step 3: Tensors / device
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)

# ===========================
# Step 4: Define MLP
# ===========================
class MLP(nn.Module):
    def __init__(self, input_size, output_size=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )
    def forward(self, x):
        return self.net(x)

model = MLP(input_size=X_train_tensor.shape[1], output_size=y_train_tensor.shape[1]).to(device)

# ===========================
# Step 5: Train
# ===========================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 500
batch_size = 64

model.train()
for epoch in range(epochs):
    perm = torch.randperm(X_train_tensor.size(0), device=device)
    for i in range(0, X_train_tensor.size(0), batch_size):
        idx = perm[i:i+batch_size]
        xb = X_train_tensor[idx]
        yb = y_train_tensor[idx]

        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

    if epoch % 50 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")

# ===========================
# Step 6: Evaluate
# ===========================
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor).cpu().numpy()

y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_unscaled = scaler_y.inverse_transform(y_test_tensor.cpu().numpy())

mae = mean_absolute_error(y_test_unscaled, y_pred, multioutput="raw_values")
r2 = r2_score(y_test_unscaled, y_pred, multioutput="raw_values")

print("\n📊 MLP Performance (on Test Set):")
for i, lbl in enumerate(label_names):
    print(f"MAE - {lbl:5s}: {mae[i]:.4f}")
for i, lbl in enumerate(label_names):
    print(f"R²  - {lbl:5s}: {r2[i]:.4f}")

# ===========================
# Step 7: Plots
# ===========================
for i, lbl in enumerate(label_names):
    plt.figure(figsize=(4.5, 4))
    plt.scatter(y_test_unscaled[:, i], y_pred[:, i], alpha=0.7)
    plt.xlabel(f"True {lbl}")
    plt.ylabel(f"Predicted {lbl}")
    plt.title(f"MLP Prediction: {lbl}")
    plt.grid(True)
    plt.axis("equal")
    min_val, max_val = np.min(y_test_unscaled[:, i]), np.max(y_test_unscaled[:, i])
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"mlp_prediction_{lbl}.png"), dpi=300)
    plt.close()

print(f"✅ MLP training complete. Results saved in {OUT_DIR}/")

# ===========================
# Step 8: Spatial force error map (for intuition)
# ===========================
x_coords = y_test_unscaled[:, 0]
y_coords = y_test_unscaled[:, 1]
force_true = y_test_unscaled[:, 2]
force_pred = y_pred[:, 2]
force_error = np.abs(force_pred - force_true)

plt.figure(figsize=(6, 6))
sc = plt.scatter(x_coords, y_coords, c=force_error, cmap='hot', s=60, edgecolors='k')
plt.colorbar(sc, label="Force prediction error (N)")
plt.xlabel("x [m]" if DATA_SOURCE == "real" else "x (sim)")
plt.ylabel("y [m]" if DATA_SOURCE == "real" else "y (sim)")
plt.title("MLP Force Prediction Error by Touch Location")
plt.gca().set_aspect("equal")
plt.grid(True)
plt.savefig(os.path.join(OUT_DIR, "force_error_map.png"), dpi=300)
plt.close()
print(f"📍 Saved spatial force error heatmap → {os.path.join(OUT_DIR, 'force_error_map.png')}")
