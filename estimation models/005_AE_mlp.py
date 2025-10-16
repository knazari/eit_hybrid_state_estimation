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
    OUT_DIR = "results_ae_mlp_sim"
elif DATA_SOURCE == "real":
    # Point to one real CSV (or a merged CSV of multiple trials)
    DATA_PATH = "data/merged_with_probe.csv"
    OUT_DIR = "results_ae_mlp_real"
else:
    raise ValueError("DATA_SOURCE must be 'sim' or 'real'")

os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===========================
# Load & prepare
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
print(f"Dataset: {DATA_SOURCE} | Samples: {X.shape[0]} | Features: {n_features}")

# ===========================
# Split & scale
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
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
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
X_test_tensor  = torch.tensor(X_test_scaled,  dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
y_test_tensor  = torch.tensor(y_test_scaled,  dtype=torch.float32).to(device)

# ===========================
# Models
# ===========================
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
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

class Regressor(nn.Module):
    def __init__(self, latent_dim=16, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )
    def forward(self, z):
        return self.net(z)

latent_dim = 16
autoencoder = Autoencoder(input_dim=n_features, latent_dim=latent_dim).to(device)
regressor   = Regressor(latent_dim=latent_dim, output_dim=y_train_tensor.shape[1]).to(device)

# Directory to save the best model
os.makedirs("models", exist_ok=True)
best_model_path = "models/AEmlp_model.pth"

# ===========================
# Train Autoencoder
# ===========================
ae_optimizer = optim.Adam(autoencoder.parameters(), lr=1e-3)
ae_loss_fn = nn.MSELoss()
ae_epochs = 500
ae_batch = 256

print("\n🔧 Training Autoencoder...")
autoencoder.train()
for epoch in range(ae_epochs):
    perm = torch.randperm(X_train_tensor.size(0), device=device)
    epoch_loss = 0.0
    for i in range(0, X_train_tensor.size(0), ae_batch):
        idx = perm[i:i+ae_batch]
        xb = X_train_tensor[idx]
        x_recon = autoencoder(xb)
        loss = ae_loss_fn(x_recon, xb)
        ae_optimizer.zero_grad()
        loss.backward()
        ae_optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    if epoch % 50 == 0 or epoch == ae_epochs - 1:
        print(f"AE Epoch {epoch:4d} | Loss: {epoch_loss / X_train_tensor.size(0):.6f}")

# ===========================
# Freeze encoder & extract latents
# ===========================
autoencoder.eval()
with torch.no_grad():
    Z_train = autoencoder.encoder(X_train_tensor)
    Z_test  = autoencoder.encoder(X_test_tensor)

# ===========================
# Train regressor on latents
# ===========================
reg_optimizer = optim.Adam(regressor.parameters(), lr=1e-3)
reg_loss_fn = nn.MSELoss()
reg_epochs = 1000
reg_batch = 256
best_loss = float("inf")


print("\n🔧 Training Transformer...")
for epoch in range(reg_epochs):
    regressor.train()
    perm = torch.randperm(Z_train.size(0))
    epoch_loss = 0

    for i in range(0, Z_train.size(0), reg_batch):
        idx = perm[i:i+reg_batch]
        x_batch = Z_train[idx].to(device)
        y_batch = y_train_tensor[idx].to(device)

        reg_optimizer.zero_grad()
        y_pred = regressor(x_batch)
        loss = reg_loss_fn(y_pred, y_batch)
        loss.backward()
        reg_optimizer.step()

        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / (Z_train.size(0) // reg_batch)

    # Save best regressor
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        torch.save(regressor.state_dict(), best_model_path)

    if epoch % 50 == 0 or epoch == reg_epochs - 1:
        print(f"Epoch {epoch:3d} | Avg Loss: {avg_epoch_loss:.6f} | Best Loss: {best_loss:.6f}")

# ===========================
# Evaluate
# ===========================
autoencoder.eval(); regressor.eval()
with torch.no_grad():
    y_pred_scaled = regressor(Z_test).detach().cpu().numpy()

y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_unscaled = scaler_y.inverse_transform(y_test_tensor.detach().cpu().numpy())

mae = mean_absolute_error(y_test_unscaled, y_pred, multioutput="raw_values")
r2  = r2_score(y_test_unscaled, y_pred, multioutput="raw_values")

print("\n📊 Autoencoder + MLP Performance:")
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
    plt.title(f"AE+MLP Prediction: {lbl}")
    plt.grid(True)
    plt.axis("equal")
    lo, hi = float(np.min(y_test_unscaled[:, i])), float(np.max(y_test_unscaled[:, i]))
    plt.plot([lo, hi], [lo, hi], 'r--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"ae_mlp_prediction_{lbl}.png"), dpi=300)
    plt.close()

print(f"✅ Training complete. Plots saved in '{OUT_DIR}/'")
