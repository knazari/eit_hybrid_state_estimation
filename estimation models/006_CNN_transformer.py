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


plt.rcParams['pdf.fonttype'] = 42   # use TrueType fonts, not Type 3
plt.rcParams['ps.fonttype'] = 42


# ===========================
# Config: choose dataset
# ===========================
DATA_SOURCE = "real"   # "sim" or "real"

if DATA_SOURCE == "sim":
    DATA_PATH = "/home/kiyanoush/Projects/eit_hybrid_state_estimation/data/touch_force_dataset_sim2real.csv"
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

# ===========================
# Train/test split & scaling
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

# Tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)[:, None, :].to(device)  # (N,1,L)
X_test_tensor  = torch.tensor(X_test_scaled,  dtype=torch.float32)[:, None, :].to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
y_test_tensor  = torch.tensor(y_test_scaled,  dtype=torch.float32).to(device)

# ===========================
# Model
# ===========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)
    def forward(self, x):  # x: (B, L, d_model)
        return x + self.pe[:, :x.size(1), :]

class CNNTransformerRegressor(nn.Module):
    def __init__(self, input_len, d_model=64, nhead=8, num_layers=2, out_dim=3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),          # L -> floor(L/2)
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),          # L -> floor(L/4)
        )
        # Determine post-CNN sequence length dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_len)
            cnn_out = self.cnn(dummy)          # (1, 64, L')
            self.seq_len = cnn_out.shape[-1]   # L'
        self.proj = nn.Linear(64, d_model)
        self.pos = PositionalEncoding(d_model=d_model, max_len=self.seq_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )
    def forward(self, x):                 # x: (B,1,L)
        x = self.cnn(x)                   # (B,64,L')
        x = x.permute(0, 2, 1)            # (B,L',64)
        x = self.proj(x)                  # (B,L',d_model)
        x = self.pos(x)                   # (B,L',d_model)
        x = self.encoder(x)               # (B,L',d_model)
        x = x.mean(dim=1)                 # (B,d_model)
        return self.regressor(x)          # (B,3)

model = CNNTransformerRegressor(input_len=n_features, out_dim=y_train_tensor.shape[1]).to(device)

# Directory to save the best model
os.makedirs("models", exist_ok=True)
best_model_path = "models/CNNtransformer_model.pth"

# ===========================
# Train
# ===========================
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
loss_fn = nn.MSELoss()
epochs = 500
batch_size = 64
best_loss = float("inf")

print("\n🔧 Training Transformer...")
for epoch in range(epochs):
    model.train()
    perm = torch.randperm(X_train_tensor.size(0))
    epoch_loss = 0

    for i in range(0, X_train_tensor.size(0), batch_size):
        idx = perm[i:i+batch_size]
        x_batch = X_train_tensor[idx].to(device)
        y_batch = y_train_tensor[idx].to(device)

        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / (X_train_tensor.size(0) // batch_size)

    # Save best model
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        torch.save(model.state_dict(), best_model_path)

    if epoch % 50 == 0 or epoch == epochs - 1:
        print(f"Epoch {epoch:3d} | Avg Loss: {avg_epoch_loss:.6f} | Best Loss: {best_loss:.6f}")

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

print("\n📊 CNN+Transformer Performance:")
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
    plt.xlabel(f"True {lbl}", fontsize=16)
    plt.ylabel(f"Predicted {lbl}", fontsize=16)
    # plt.title(f"CNN+Transformer: {lbl}")
    # plt.grid(True)
    plt.axis("equal")
    lo, hi = float(np.min(y_test_unscaled[:, i])), float(np.max(y_test_unscaled[:, i]))
    plt.plot([lo, hi], [lo, hi], 'r--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"cnntr_pred_{lbl}.pdf"), dpi=300, bbox_inches="tight")
    plt.close()

print(f"✅ Training complete. Plots saved in '{OUT_DIR}/'")


# ===========================
# Step 8: Spatial force error map (for intuition)
# ===========================
x_coords = y_test_unscaled[:, 0]
y_coords = y_test_unscaled[:, 1]
force_true = y_test_unscaled[:, 2]
force_pred = y_pred[:, 2]
force_error = np.abs(force_pred - force_true)

plt.figure(figsize=(6.5, 6.6))
sc = plt.scatter(x_coords, y_coords, c=force_error, cmap='hot', s=60, edgecolors='k')
plt.colorbar(sc, label="Force prediction error (N)")
plt.xlabel("x [m]" if DATA_SOURCE == "real" else "x (sim)", fontsize=16)
plt.ylabel("y [m]" if DATA_SOURCE == "real" else "y (sim)", fontsize=16)
# plt.title("MLP Force Prediction Error by Touch Location")
plt.gca().set_aspect("equal")
# plt.grid(True)
plt.savefig(os.path.join(OUT_DIR, "force_error_map.pdf"), dpi=300, bbox_inches="tight")
plt.close()
print(f"📍 Saved spatial force error heatmap → {os.path.join(OUT_DIR, 'force_error_map.pdf')}")