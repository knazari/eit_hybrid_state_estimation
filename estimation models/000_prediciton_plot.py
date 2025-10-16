# ==========================================
# 000_plot_cnn_transformer.py
# Plot predictions for trained CNN+Transformer
# ==========================================
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ===========================
# Config
# ===========================
DATA_SOURCE = "real"   # "sim" or "real"
if DATA_SOURCE == "sim":
    DATA_PATH = "/home/kiyanoush/Projects/eit_hybrid_state_estimation/data/touch_force_dataset_sim2real.csv"
    OUT_DIR = "results_cnntr_sim"
elif DATA_SOURCE == "real":
    DATA_PATH = "data/merged_with_probe.csv"
    OUT_DIR = "results_cnntr_real"
else:
    raise ValueError("DATA_SOURCE must be 'sim' or 'real'")

os.makedirs(OUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================
# Load dataset
# ===========================
df = pd.read_csv(DATA_PATH)

if DATA_SOURCE == "sim":
    X = df.iloc[:, :-3].values
    y = df.iloc[:, -3:].values
    label_names = ["x", "y", "force"]
else:
    eit_cols = [c for c in df.columns if c.startswith("eit_") and c.lower() != "t_eit"]
    eps = 1e-9
    variable_eit_cols = [c for c in eit_cols if df[c].std(skipna=True) is not None and df[c].std() > eps]
    targets = df[["contact_u_m", "contact_v_m", "force_n"]].copy()
    mask_contact   = targets["force_n"].notna() & (targets["force_n"] >= 0.5)
    mask_baseline  = targets["force_n"].notna() & (targets["force_n"] < 0.05)
    v_ref = df.loc[mask_baseline, variable_eit_cols].astype(np.float32).to_numpy()
    v_ref = np.nanmean(v_ref, axis=0, dtype=np.float64).astype(np.float32)
    X_raw = df.loc[mask_contact, variable_eit_cols].astype(np.float32).to_numpy()
    X = (X_raw - v_ref[None, :]).astype(np.float32)
    y = targets.loc[mask_contact, :].to_numpy(dtype=np.float32)
    label_names = ["x", "y", "force"]

n_features = X.shape[1]

# ===========================
# Split + scaling (must match training!)
# ===========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled  = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled  = scaler_y.transform(y_test)

X_test_tensor  = torch.tensor(X_test_scaled, dtype=torch.float32)[:, None, :].to(device)
y_test_tensor  = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)

# ===========================
# Model definition (same as training)
# ===========================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class CNNTransformerRegressor(nn.Module):
    def __init__(self, input_len, d_model=64, nhead=8, num_layers=2, out_dim=3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_len)
            cnn_out = self.cnn(dummy)
            self.seq_len = cnn_out.shape[-1]
        self.proj = nn.Linear(64, d_model)
        self.pos = PositionalEncoding(d_model, self.seq_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=0.1, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Linear(64, out_dim)
        )
    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        x = self.pos(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.regressor(x)

# Load trained model
model = CNNTransformerRegressor(input_len=n_features, out_dim=y_train.shape[1]).to(device)
model.load_state_dict(torch.load("models/CNNtransformer_model.pth", map_location=device))
model.eval()

# ===========================
# Inference
# ===========================
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor).cpu().numpy()

y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_unscaled = scaler_y.inverse_transform(y_test_tensor.cpu().numpy())

# ===========================
# Plots (scatter parity)
# ===========================
for i, lbl in enumerate(label_names):
    plt.figure(figsize=(4.5, 4))
    plt.scatter(y_test_unscaled[:, i], y_pred[:, i], alpha=0.7)
    plt.xlabel(f"True {lbl}", fontsize=16)
    plt.ylabel(f"Predicted {lbl}", fontsize=16)
    plt.axis("equal")
    lo, hi = float(np.min(y_test_unscaled[:, i])), float(np.max(y_test_unscaled[:, i]))
    plt.plot([lo, hi], [lo, hi], 'r--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"cnntr_pred_{lbl}.png"), dpi=300)
    plt.close()

print(f"✅ Done. Prediction plots saved to '{OUT_DIR}/'")
