# ============================================================
# phy_latent_cnn.py
# ------------------------------------------------------------
# Physics-informed latent regularizer, CNN version.
# Same preprocessing as your MLP script.
# Stage A:
#   • VoltAE: Enc_v (CNN) : v_s -> z, Dec_v (MLP) : z -> v_s
#   • ZReg: (x,y,F)_s -> z  (teacher = Enc_v(v_s))
# Stage B:
#   • F_cnn: v_s -> (x,y,F)_s
#   • Loss = L_sup + λ * || ZReg(y_pred) - Enc_v(v_s) ||^2
# Includes eval + parity plots + spatial force error map.
# ============================================================

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
# Config
# ===========================
DATA_SOURCE = "sim"   # "sim" or "real"
if DATA_SOURCE == "sim":
    DATA_PATH = "/home/kiyanoush/Projects/eit_hybrid_state_estimation/data/touch_force_dataset_sim2real.csv"
    OUT_DIR = "results_phy_latent_sim_cnn"
else:
    DATA_PATH = "data/merged_with_probe.csv"
    OUT_DIR = "results_phy_latent_real_cnn"

os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Training knobs
EPOCHS_F   = 500
BATCH_F    = 64
LR_F       = 1e-3
WD_F       = 0.0
FORCE_W    = 1.0  # keep equal weights (change if you want to upweight force)

# Latent regularizer
LAMBDA_LATENT = 0.03
Z_DIM         = 16

# Stage A (VoltAE & ZReg) knobs
EPOCHS_AE = 300
EPOCHS_ZR = 300
BATCH_A   = 128
LR_AE     = 1e-3
LR_ZR     = 1e-3
WD_AE     = 1e-4
WD_ZR     = 1e-4

# ===========================
# Step 1: Load and split (same as your baseline)
# ===========================
df = pd.read_csv(DATA_PATH)

if DATA_SOURCE == "sim":
    # Simulated: last 3 columns = x, y, force
    X = df.iloc[:, :-3].values.astype(np.float32)
    y = df.iloc[:, -3:].values.astype(np.float32)
    label_names = ["x", "y", "force"]
else:
    # Real: baseline subtraction
    eit_cols = [c for c in df.columns if c.startswith("eit_") and c.lower() != "t_eit"]
    if len(eit_cols) == 0:
        raise RuntimeError("No EIT columns found.")
    eps = 1e-9
    variable_eit_cols = []
    for c in eit_cols:
        std_c = df[c].std(skipna=True)
        if df[c].nunique(dropna=False) > 1 and std_c is not None and std_c > eps:
            variable_eit_cols.append(c)
    if len(variable_eit_cols) == 0:
        raise RuntimeError("All EIT channels are near-constant.")

    targets = df[["contact_u_m", "contact_v_m", "force_n"]].copy()
    mask_contact   = targets["force_n"].notna() & (targets["force_n"] >= 0.5)
    mask_baseline  = targets["force_n"].notna() & (targets["force_n"] <  0.05)

    if mask_baseline.sum() == 0:
        raise RuntimeError("No 'no-touch' rows (force_n < 0.05) to compute baseline.")

    v_ref = df.loc[mask_baseline, variable_eit_cols].astype(np.float32).to_numpy()
    v_ref = np.nanmean(v_ref, axis=0, dtype=np.float64).astype(np.float32)  # [D]

    X_raw = df.loc[mask_contact, variable_eit_cols].astype(np.float32).to_numpy()
    X = (X_raw - v_ref[None, :]).astype(np.float32)
    y = targets.loc[mask_contact, :].to_numpy(dtype=np.float32)
    label_names = ["x", "y", "force"]

print(f"Dataset: {DATA_SOURCE} | Samples: {X.shape[0]} | Features: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===========================
# Step 2: Standardize
# ===========================
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_s = scaler_X.fit_transform(X_train).astype(np.float32)
X_test_s  = scaler_X.transform(X_test).astype(np.float32)
y_train_s = scaler_y.fit_transform(y_train).astype(np.float32)
y_test_s  = scaler_y.transform(y_test).astype(np.float32)

# ===========================
# Step 3: Tensors (note CNN needs channel dim)
# ===========================
Xtr = torch.tensor(X_train_s, dtype=torch.float32, device=device)
Xte = torch.tensor(X_test_s,  dtype=torch.float32, device=device)
Ytr = torch.tensor(y_train_s, dtype=torch.float32, device=device)
Yte = torch.tensor(y_test_s,  dtype=torch.float32, device=device)

Xtr_cnn = Xtr[:, None, :]  # [N,1,L]
Xte_cnn = Xte[:, None, :]  # [N,1,L]

# ===========================
# Step 4: Models
# ===========================
class CNNVoltEncoder(nn.Module):
    """v_s [N,1,L] -> z [N,Z_DIM]"""
    def __init__(self, input_len, z_dim=Z_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_len)
            flat = self.conv(dummy).view(1, -1).shape[1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 256), nn.ReLU(),
            nn.Linear(256, z_dim)
        )
    def forward(self, x):  # x: [N,1,L]
        return self.head(self.conv(x))

class VoltDecoder(nn.Module):
    """z -> v_s [N,L] via MLP (simple + stable)"""
    def __init__(self, z_dim=Z_DIM, out_dim=None):
        super().__init__()
        assert out_dim is not None
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, out_dim)
        )
    def forward(self, z):  # z: [N,Z]
        return self.net(z)

class ZReg(nn.Module):
    """(x,y,F)_s -> z"""
    def __init__(self, in_dim=3, z_dim=Z_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, z_dim)
        )
    def forward(self, y): return self.net(y)

class CNNPredictor(nn.Module):
    """v_s [N,1,L] -> (x,y,F)_s [N,3]"""
    def __init__(self, input_len, out_dim=3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_len)
            flat = self.conv(dummy).view(1, -1).shape[1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x):  # x: [N,1,L]
        return self.head(self.conv(x))

mse = nn.MSELoss()

# ===========================
# Stage A1: VoltAE (Enc_v: CNN, Dec_v: MLP)
# ===========================
Enc_v = CNNVoltEncoder(input_len=Xtr.shape[1], z_dim=Z_DIM).to(device)
Dec_v = VoltDecoder(z_dim=Z_DIM, out_dim=Xtr.shape[1]).to(device)
opt_AE = optim.Adam(list(Enc_v.parameters()) + list(Dec_v.parameters()),
                    lr=LR_AE, weight_decay=WD_AE)

print("\n=== Stage A1: train VoltAE (v_s <-> z) ===")
N = Xtr_cnn.size(0)
for ep in range(EPOCHS_AE):
    Enc_v.train(); Dec_v.train()
    perm = torch.randperm(N, device=device)
    ep_loss = 0.0
    for i in range(0, N, BATCH_A):
        idx = perm[i:i+BATCH_A]
        vb = Xtr_cnn[idx]                 # [B,1,L]
        zb = Enc_v(vb)                    # [B,Z]
        vb_hat = Dec_v(zb)                # [B,L]
        loss = mse(vb_hat, vb.squeeze(1)) # compare to [B,L]
        opt_AE.zero_grad(); loss.backward(); opt_AE.step()
        ep_loss += float(loss.item()) * vb.size(0)
    ep_loss /= N

    if ep % 25 == 0 or ep == EPOCHS_AE-1:
        Enc_v.eval(); Dec_v.eval()
        with torch.no_grad():
            val = mse(Dec_v(Enc_v(Xte_cnn)), Xte_cnn.squeeze(1)).item()
        print(f"[VoltAE] Ep {ep:03d} | train {ep_loss:.6f} | val_rec {val:.6f}")

# Freeze Enc_v for Stage B (we still use it, but not training)
for p in Dec_v.parameters(): p.requires_grad = False
Dec_v.eval()
for p in Enc_v.parameters(): p.requires_grad = False
Enc_v.eval()

# Teacher latents
with torch.no_grad():
    Ztr_teacher = Enc_v(Xtr_cnn)  # [N,Z]
    Zte_teacher = Enc_v(Xte_cnn)

# ===========================
# Stage A2: ZReg: (x,y,F)_s -> z_v
# ===========================
zreg = ZReg(in_dim=Ytr.shape[1], z_dim=Z_DIM).to(device)
opt_ZR = optim.Adam(zreg.parameters(), lr=LR_ZR, weight_decay=WD_ZR)

print("\n=== Stage A2: train ZReg ((x,y,F)_s -> z_v) ===")
N = Ytr.size(0)
for ep in range(EPOCHS_ZR):
    zreg.train()
    perm = torch.randperm(N, device=device)
    ep_loss = 0.0
    for i in range(0, N, BATCH_A):
        idx = perm[i:i+BATCH_A]
        yb = Ytr[idx]               # [B,3]
        zt = Ztr_teacher[idx]       # [B,Z]
        zhat = zreg(yb)             # [B,Z]
        loss = mse(zhat, zt)
        opt_ZR.zero_grad(); loss.backward(); opt_ZR.step()
        ep_loss += float(loss.item()) * yb.size(0)
    ep_loss /= N

    if ep % 25 == 0 or ep == EPOCHS_ZR-1:
        zreg.eval()
        with torch.no_grad():
            val = mse(zreg(Yte), Zte_teacher).item()
        print(f"[ZReg] Ep {ep:03d} | train {ep_loss:.6f} | val {val:.6f}")

# Freeze zreg for Stage B
for p in zreg.parameters(): p.requires_grad = False
zreg.eval()

# ===========================
# Stage B: CNN predictor with latent regularizer
# ===========================
F_cnn = CNNPredictor(input_len=Xtr.shape[1], out_dim=3).to(device)
opt_F = optim.Adam(F_cnn.parameters(), lr=LR_F, weight_decay=WD_F)

print("\n=== Stage B: train CNN (v_s -> (x,y,F)_s) with latent regularizer ===")
best_val = float("inf")
N = Xtr_cnn.size(0)

for ep in range(EPOCHS_F):
    F_cnn.train()
    perm = torch.randperm(N, device=device)
    ep_loss = 0.0
    for i in range(0, N, BATCH_F):
        idx = perm[i:i+BATCH_F]
        vb = Xtr_cnn[idx]   # [B,1,L]
        yb = Ytr[idx]       # [B,3]

        opt_F.zero_grad()
        y_pred = F_cnn(vb)  # [B,3]

        # supervised (optionally upweight force)
        loss_sup = mse(y_pred[:, :2], yb[:, :2]) + FORCE_W * mse(y_pred[:, 2:], yb[:, 2:])

        # latent consistency: ZReg(y_pred) ≈ Enc_v(vb)
        z_from_y = zreg(y_pred)
        with torch.no_grad():  # teacher is fixed
            z_from_v = Enc_v(vb)
        loss_lat = mse(z_from_y, z_from_v)

        loss = loss_sup + LAMBDA_LATENT * loss_lat
        loss.backward(); opt_F.step()
        ep_loss += float(loss.item()) * vb.size(0)
    ep_loss /= N

    if (ep % 25) == 0 or ep == EPOCHS_F - 1:
        F_cnn.eval()
        with torch.no_grad():
            y_val   = F_cnn(Xte_cnn)
            sup_val = (mse(y_val[:, :2], Yte[:, :2]) + FORCE_W * mse(y_val[:, 2:], Yte[:, 2:])).item()
            lat_v   = mse(zreg(y_val), Enc_v(Xte_cnn)).item()
            val_tot = sup_val + LAMBDA_LATENT * lat_v

        print(f"[StageB] Ep {ep:03d} | train {ep_loss:.6f} | val_sup {sup_val:.6f} | val_lat {lat_v:.6f} | val_total {val_tot:.6f}")

        if val_tot < best_val:
            best_val = val_tot
            torch.save({
                "F_state_dict": F_cnn.state_dict(),
                "Enc_v": Enc_v.state_dict(),
                "ZReg": zreg.state_dict(),
                "scaler_X_mean": scaler_X.mean_,
                "scaler_X_scale": scaler_X.scale_,
                "scaler_y_mean": scaler_y.mean_,
                "scaler_y_scale": scaler_y.scale_,
                "z_dim": Z_DIM,
            }, os.path.join(OUT_DIR, "best_latent_model_cnn.pt"))
            print(f"  ↳ ✅ new best total={best_val:.6f}; saved to {OUT_DIR}/best_latent_model_cnn.pt")

# ===========================
# Evaluation (same as baseline)
# ===========================
F_cnn.eval()
with torch.no_grad():
    y_pred_s = F_cnn(Xte_cnn).cpu().numpy()

y_pred = scaler_y.inverse_transform(y_pred_s)
y_true = scaler_y.inverse_transform(Yte.cpu().numpy())

mae = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
r2  = r2_score(y_true, y_pred, multioutput="raw_values")

print("\n📊 Physics-informed (latent) CNN — Test Set")
for i, lbl in enumerate(["x","y","force"]):
    print(f"MAE - {lbl:5s}: {mae[i]:.4f}")
for i, lbl in enumerate(["x","y","force"]):
    print(f"R²  - {lbl:5s}: {r2[i]:.4f}")

# ===========================
# Plots
# ===========================
for i, lbl in enumerate(["x","y","force"]):
    plt.figure(figsize=(4.5, 4))
    plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.7)
    plt.xlabel(f"True {lbl}")
    plt.ylabel(f"Predicted {lbl}")
    plt.title(f"Physics-informed (latent) CNN: {lbl}")
    plt.grid(True)
    plt.axis("equal")
    min_val, max_val = np.min(y_true[:, i]), np.max(y_true[:, i])
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"phy_latent_cnn_prediction_{lbl}.png"), dpi=300)
    plt.close()

# Spatial force error map
x_coords = y_true[:, 0]
y_coords = y_true[:, 1]
force_true = y_true[:, 2]
force_pred = y_pred[:, 2]
force_error = np.abs(force_pred - force_true)

plt.figure(figsize=(6, 6))
sc = plt.scatter(x_coords, y_coords, c=force_error, cmap='hot', s=60, edgecolors='k')
plt.colorbar(sc, label="Force prediction error")
plt.xlabel("x (sim)" if DATA_SOURCE == "sim" else "x [m]")
plt.ylabel("y (sim)" if DATA_SOURCE == "sim" else "y [m]")
plt.title("Physics-informed CNN — Force Error by Location")
plt.gca().set_aspect("equal")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "phy_latent_force_error_map_cnn.png"), dpi=300)
plt.close()

print(f"\n✅ Done. Outputs in: {OUT_DIR}/")
