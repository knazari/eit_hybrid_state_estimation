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
    OUT_DIR = "results_transformer_sim"
elif DATA_SOURCE == "real":
    DATA_PATH = "data/merged_with_probe.csv"  # <- set to your CSV (or merged CSV)
    OUT_DIR = "results_transformer_real"
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


seq_len = X.shape[1]                    # transformer sequence length = #EIT channels
print(f"Dataset: {DATA_SOURCE} | Samples: {X.shape[0]} | Seq len: {seq_len}")

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

# Tensors: Transformer expects (B, Seq, Features). Here features=1.
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(-1).to(device)  # (N, L, 1)
X_test_tensor  = torch.tensor(X_test_scaled,  dtype=torch.float32).unsqueeze(-1).to(device)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
y_test_tensor  = torch.tensor(y_test_scaled,  dtype=torch.float32).to(device)

# ===========================
# Model
# ===========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):  # x: (B, L, d_model)
        return x + self.pe[:, :x.size(1), :]

class TransformerRegressor(nn.Module):
    def __init__(self, seq_len, d_model=64, nhead=8, num_layers=2, out_dim=3):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):                # x: (B, L, 1)
        x = self.input_proj(x)           # (B, L, d_model)
        x = self.pos_encoder(x)          # (B, L, d_model)
        x = self.encoder(x)              # (B, L, d_model)
        x = x.mean(dim=1)                # global average pooling over L
        return self.regressor(x)         # (B, out_dim)

model = TransformerRegressor(seq_len=seq_len, out_dim=y_train_tensor.shape[1]).to(device)

# Directory to save the best model
os.makedirs("models", exist_ok=True)
best_model_path = "models/transformer_model.pth"

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

print("\n📊 Transformer Model Performance:")
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
    plt.title(f"Transformer Prediction: {lbl}")
    plt.grid(True)
    plt.axis("equal")
    lo, hi = float(np.min(y_test_unscaled[:, i])), float(np.max(y_test_unscaled[:, i]))
    plt.plot([lo, hi], [lo, hi], 'r--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"transformer_pred_{lbl}.png"), dpi=300)
    plt.close()

print(f"✅ Training complete. Plots saved in '{OUT_DIR}/'")
