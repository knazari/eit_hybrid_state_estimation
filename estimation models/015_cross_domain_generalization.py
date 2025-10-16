# cross_domain_eval.py
# ------------------------------------------------------------
# Sim -> Real cross-domain evaluation with:
#   - Zero-shot (sim-scaled)
#   - CORAL alignment (unsupervised; uses few-shot real for stats)
#   - CORAL + Affine Adapter (few-shot supervised)
# Optional:
#   - Quantile matching (unsupervised) before CORAL
#   - Force head calibration (few-shot isotonic/affine)
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ===========================
# User config
# ===========================
SIM_CSV   = "/home/kiyanoush/Projects/eit_hybrid_state_estimation/data/touch_force_dataset_sim2real.csv"
REAL_CSV  = "data/merged_with_probe.csv"
OUT_DIR   = "xdom_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RNG_SEED = 42
np.random.seed(RNG_SEED); torch.manual_seed(RNG_SEED)

# Model/adapter knobs
MODEL_TYPE = "mlp"      # "mlp" or "cnn1d"
LOAD_MODEL_PATH = None  # e.g., "models/sim_trained_mlp.pth" (if None -> train here)
EPOCHS_SIM = 300
BATCH_SIM  = 128
LR_SIM     = 1e-3
WD_SIM     = 1e-4

# Few-shot real percentages to report (choose one at a time for adapter/calibration)
FEWSHOT_PCT = 0.05      # 0.01 or 0.10 etc.

# Domain alignment toggles
USE_QMATCH   = False    # optional; per-channel quantile matching before CORAL
USE_CORAL    = True     # mean+cov alignment
USE_ADAPTER  = True     # learn tiny affine after CORAL (few-shot supervised)
ADAPTER_LR   = 5e-3
ADAPTER_STEPS= 400
ADAPTER_BATCH= 128

# Optional force calibration (few-shot real)
CALIBRATE_FORCE = True  # learns mapping on predicted force -> true force (few-shot)
CALIB_METHOD    = "isotonic"  # "isotonic" or "affine"

# Real-data preprocessing constants
FORCE_CONTACT_MIN = 0.5
FORCE_BASELINE_MAX = 0.05
EIT_PREFIX = "eit_"

# ===========================
# Data loading helpers
# ===========================
def load_sim(csv_path):
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-3].to_numpy(dtype=np.float32)
    y = df.iloc[:, -3:].to_numpy(dtype=np.float32)
    return X, y, ["x","y","force"]

def load_real(csv_path):
    df = pd.read_csv(csv_path)
    # select EIT columns (variable)
    eit_cols = [c for c in df.columns if c.startswith(EIT_PREFIX) and c.lower() != "t_eit"]
    if not eit_cols:
        raise RuntimeError("No EIT columns starting with 'eit_' found.")
    eps = 1e-9
    variable_eit = [c for c in eit_cols
                    if (df[c].nunique(dropna=False) > 1) and (df[c].std(skipna=True) is not None) and (df[c].std(skipna=True) > eps)]

    tgt = df[["contact_u_m","contact_v_m","force_n"]].copy()
    mask_contact  = tgt["force_n"].notna() & (tgt["force_n"] >= FORCE_CONTACT_MIN)
    mask_baseline = tgt["force_n"].notna() & (tgt["force_n"] <  FORCE_BASELINE_MAX)
    if mask_baseline.sum() == 0:
        raise RuntimeError("No 'no-touch' rows to compute baseline (force_n < 0.05).")

    v_ref = df.loc[mask_baseline, variable_eit].astype(np.float32).to_numpy()
    v_ref = np.nanmean(v_ref, axis=0, dtype=np.float64).astype(np.float32)

    X_raw = df.loc[mask_contact, variable_eit].astype(np.float32).to_numpy()
    X = (X_raw - v_ref[None, :]).astype(np.float32)
    y = tgt.loc[mask_contact, :].to_numpy(dtype=np.float32)
    return X, y, ["x","y","force"]

# ===========================
# Models
# ===========================
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, out_dim),
        )
    def forward(self, x):  # x: (N, D) or (N,1,L) flattened by caller
        return self.net(x)

class CNN1D_Head(nn.Module):
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
            flat = self.conv(dummy).view(1,-1).shape[1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, out_dim),
        )
    def forward(self, x):  # x: (N,1,L)
        return self.head(self.conv(x))

class AffineAdapter(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(D))
        self.beta  = nn.Parameter(torch.zeros(D))
    def forward(self, X):  # (N,D)
        return self.gamma * X + self.beta

# ===========================
# CORAL & Quantile matching
# ===========================
def _symm_matrix_sqrt(M, eps=1e-5):
    w, V = np.linalg.eigh(M + eps*np.eye(M.shape[0], dtype=M.dtype))
    return (V * np.sqrt(np.clip(w, 0, None))) @ V.T

def coral_fit(Xs, Xr):
    mu_s = Xs.mean(axis=0, keepdims=True)
    mu_r = Xr.mean(axis=0, keepdims=True)
    Cs = np.cov((Xs - mu_s).T, bias=False)
    Cr = np.cov((Xr - mu_r).T, bias=False)
    Cs_inv_sqrt = np.linalg.pinv(_symm_matrix_sqrt(Cs))
    Cr_sqrt     = _symm_matrix_sqrt(Cr)
    W = Cs_inv_sqrt @ Cr_sqrt
    return {"mu_s": mu_s.astype(np.float32),
            "mu_r": mu_r.astype(np.float32),
            "W":    W.astype(np.float32)}

def coral_transform(X, coral):
    mu_s, mu_r, W = coral["mu_s"], coral["mu_r"], coral["W"]
    return ((X - mu_s) @ W) + mu_r

def fit_quantile_map(xs, xt, nq=200):
    qs = np.linspace(0.0, 1.0, nq)
    v_s = np.quantile(xs, qs)
    v_t = np.quantile(xt, qs)
    return (v_s, v_t)

def apply_quantile_map(x, v_s, v_t):
    return np.interp(x, v_s, v_t)

def quantile_match_matrix(Xs, Xt, nq=200):
    return [fit_quantile_map(Xs[:,d], Xt[:,d], nq) for d in range(Xs.shape[1])]

def quantile_transform(X, maps):
    Xq = np.empty_like(X)
    for d,(v_s,v_t) in enumerate(maps):
        Xq[:,d] = apply_quantile_map(X[:,d], v_s, v_t)
    return Xq

# ===========================
# Training / Eval helpers
# ===========================
def train_on_sim(Xs_tr, Ys_tr, model_type="mlp", epochs=300, batch=128, lr=1e-3, wd=1e-4):
    D = Xs_tr.shape[1]
    if model_type == "mlp":
        model = MLP(D, out_dim=Ys_tr.shape[1]).to(DEVICE)
        X_tensor = torch.tensor(Xs_tr, dtype=torch.float32, device=DEVICE)
    else:
        model = CNN1D_Head(D, out_dim=Ys_tr.shape[1]).to(DEVICE)
        X_tensor = torch.tensor(Xs_tr, dtype=torch.float32, device=DEVICE)[:,None,:]
    Y_tensor = torch.tensor(Ys_tr, dtype=torch.float32, device=DEVICE)

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    mse = nn.MSELoss()
    N = X_tensor.size(0)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N, device=DEVICE)
        for i in range(0, N, batch):
            idx = perm[i:i+batch]
            xb = X_tensor[idx]
            yb = Y_tensor[idx]
            opt.zero_grad()
            pred = model(xb if model_type=="cnn1d" else xb)
            loss = mse(pred, yb)
            loss.backward(); opt.step()
    return model

def predict(model, X, model_type="mlp"):
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    if model_type == "cnn1d":
        X_t = X_t[:,None,:]
    with torch.no_grad():
        y = model(X_t).cpu().numpy()
    return y

def train_adapter(adapter, model, X_fs, Y_fs, model_type="mlp", lr=5e-3, steps=400, batch=128):
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    adapter.train()
    opt = optim.Adam(adapter.parameters(), lr=lr)
    mse = nn.MSELoss()
    X_t = torch.tensor(X_fs, dtype=torch.float32, device=DEVICE)
    Y_t = torch.tensor(Y_fs, dtype=torch.float32, device=DEVICE)
    N = X_t.size(0)
    for t in range(steps):
        idx = torch.randint(0, N, (min(batch,N),), device=DEVICE)
        xb = X_t[idx]
        yb = Y_t[idx]
        xb_adj = adapter(xb)
        xb_in  = xb_adj[:,None,:] if model_type=="cnn1d" else xb_adj
        yhat = model(xb_in)
        loss = mse(yhat, yb)
        opt.zero_grad(); loss.backward(); opt.step()
    return adapter

def force_calibration(pred_force, true_force, method="isotonic"):
    if method == "isotonic":
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(pred_force.ravel(), true_force.ravel())
        return lambda z: ir.predict(z.ravel()).reshape(-1,1)
    else:
        # affine: a*z + b
        A = np.column_stack([pred_force.ravel(), np.ones_like(pred_force.ravel())])
        a,b = np.linalg.lstsq(A, true_force.ravel(), rcond=None)[0]
        return lambda z: (a*z + b)

def report_metrics(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    r2  = r2_score(y_true, y_pred, multioutput="raw_values")
    print(f"{label:28s} | MAE: {np.array2string(mae, precision=4)} | R2: {np.array2string(r2, precision=4)}")

# ===========================
# Main
# ===========================
def main():
    print("Using device:", DEVICE)

    # 1) Load data
    X_sim, Y_sim, _ = load_sim(SIM_CSV)
    X_real_raw, Y_real, _ = load_real(REAL_CSV)

    # 2) Splits & scalers
    Xs_tr, Xs_te, Ys_tr, Ys_te = train_test_split(X_sim, Y_sim, test_size=0.2, random_state=RNG_SEED)

    # Fit scalers **on SIM data only** (domain priors)
    scX = StandardScaler().fit(Xs_tr)
    scY = StandardScaler().fit(Ys_tr)

    Xs_tr_s = scX.transform(Xs_tr).astype(np.float32)
    Xs_te_s = scX.transform(Xs_te).astype(np.float32)
    Ys_tr_s = scY.transform(Ys_tr).astype(np.float32)
    Ys_te_s = scY.transform(Ys_te).astype(np.float32)

    # Apply sim-scalers to real as well (by design)
    Xr_all_s = scX.transform(X_real_raw).astype(np.float32)
    Yr_all_s = scY.transform(Y_real).astype(np.float32)

    # Few-shot split from real (super small)
    Xr_fs_s, Xr_hold_s, Yr_fs_s, Yr_hold_s = train_test_split(
        Xr_all_s, Yr_all_s, test_size=(1.0 - FEWSHOT_PCT), random_state=RNG_SEED
    )

    print(f"Sim train: {Xs_tr_s.shape}, Real few-shot: {Xr_fs_s.shape}, Real test(all): {Xr_all_s.shape}")

    # 3) Train (or load) on SIM
    if LOAD_MODEL_PATH and Path(LOAD_MODEL_PATH).exists():
        print(f"Loading model from {LOAD_MODEL_PATH}")
        if MODEL_TYPE == "mlp":
            model = MLP(in_dim=Xs_tr_s.shape[1], out_dim=Ys_tr_s.shape[1]).to(DEVICE)
        else:
            model = CNN1D_Head(input_len=Xs_tr_s.shape[1], out_dim=Ys_tr_s.shape[1]).to(DEVICE)
        model.load_state_dict(torch.load(LOAD_MODEL_PATH, map_location=DEVICE))
    else:
        print("Training SIM model...")
        model = train_on_sim(
            Xs_tr_s, Ys_tr_s, model_type=MODEL_TYPE,
            epochs=EPOCHS_SIM, batch=BATCH_SIM, lr=LR_SIM, wd=WD_SIM
        )
        torch.save(model.state_dict(), os.path.join(OUT_DIR, f"sim_{MODEL_TYPE}.pth"))

    # 4) Baselines on REAL
    # 4a) Zero-shot (just sim-scaled)
    Y_hat_zero_s = predict(model, Xr_all_s, model_type=MODEL_TYPE)
    Y_hat_zero   = scY.inverse_transform(Y_hat_zero_s)
    Y_real       = Y_real  # unscaled GT
    report_metrics(Y_real, Y_hat_zero, "Zero-shot (Sim→Real)")

    # Optional quantile matching in sim-scaled space (real → sim-shape)
    Xr_qm = Xr_all_s
    if USE_QMATCH:
        print("Applying channel-wise quantile matching (real -> sim)...")
        qm_maps = quantile_match_matrix(Xr_all_s, Xs_tr_s)  # map real to sim stats
        Xr_qm = quantile_transform(Xr_all_s, qm_maps)

    # 4b) CORAL alignment (mean+cov; unsupervised, uses few-shot real to estimate target stats)
    if USE_CORAL:
        print("Fitting CORAL (sim -> real) using few-shot real stats...")
        coral = coral_fit(Xs_tr_s, Xr_fs_s if not USE_QMATCH else quantile_transform(Xr_fs_s, qm_maps))
        Xr_coral = coral_transform(Xr_qm, coral)
        Y_hat_coral_s = predict(model, Xr_coral, model_type=MODEL_TYPE)
        Y_hat_coral   = scY.inverse_transform(Y_hat_coral_s)
        report_metrics(Y_real, Y_hat_coral, "CORAL only (unsup)")

    # 4c) CORAL + Affine Adapter (few-shot supervised)
    if USE_CORAL and USE_ADAPTER and Xr_fs_s.shape[0] >= 4:
        Xr_fs_qm = Xr_fs_s if not USE_QMATCH else quantile_transform(Xr_fs_s, qm_maps)
        Xr_fs_c  = coral_transform(Xr_fs_qm, coral)
        adapter = AffineAdapter(D=Xr_fs_c.shape[1]).to(DEVICE)
        adapter = train_adapter(adapter, model, X_fs=Xr_fs_c, Y_fs=Yr_fs_s,
                                model_type=MODEL_TYPE, lr=ADAPTER_LR, steps=ADAPTER_STEPS, batch=ADAPTER_BATCH)
        # Inference with adapter
        Xr_all_c = coral_transform(Xr_qm, coral)
        Xr_all_ca = adapter(torch.tensor(Xr_all_c, dtype=torch.float32, device=DEVICE)).detach().cpu().numpy()
        Y_hat_ca_s = predict(model, Xr_all_ca, model_type=MODEL_TYPE)
        Y_hat_ca   = scY.inverse_transform(Y_hat_ca_s)
        report_metrics(Y_real, Y_hat_ca, "CORAL + Adapter (few-shot)")

        # 4d) Optional: force calibration on few-shot (post-hoc)
        if CALIBRATE_FORCE:
            print(f"Force calibration ({CALIB_METHOD}) on few-shot...")
            # few-shot predictions after CORAL+Adapter
            Xr_fs_ca = adapter(torch.tensor(Xr_fs_c, dtype=torch.float32, device=DEVICE)).detach().cpu().numpy()
            Y_hat_fs_s = predict(model, Xr_fs_ca, model_type=MODEL_TYPE)
            Y_hat_fs   = scY.inverse_transform(Y_hat_fs_s)

            # fit calibration on force column (index 2)
            cal = force_calibration(Y_hat_fs[:, [2]], scY.inverse_transform(Yr_fs_s)[:, [2]], method=CALIB_METHOD)
            Y_hat_ca_cal = Y_hat_ca.copy()
            Y_hat_ca_cal[:, 2:3] = cal(Y_hat_ca[:, [2]])
            report_metrics(Y_real, Y_hat_ca_cal, "CORAL+Adapter + ForceCal")

    # Save quick CSV of predictions for the best variant you computed
    best_pred = {
        "y_true_x": Y_real[:,0], "y_true_y": Y_real[:,1], "y_true_f": Y_real[:,2],
        "y_zero_x": Y_hat_zero[:,0], "y_zero_y": Y_hat_zero[:,1], "y_zero_f": Y_hat_zero[:,2],
    }
    if USE_CORAL:
        best_pred["y_coral_x"] = Y_hat_coral[:,0]; best_pred["y_coral_y"] = Y_hat_coral[:,1]; best_pred["y_coral_f"] = Y_hat_coral[:,2]
    if USE_CORAL and USE_ADAPTER:
        best_pred["y_coral_adapter_x"] = Y_hat_ca[:,0]; best_pred["y_coral_adapter_y"] = Y_hat_ca[:,1]; best_pred["y_coral_adapter_f"] = Y_hat_ca[:,2]
        if CALIBRATE_FORCE:
            best_pred["y_coral_adapter_cal_f"] = Y_hat_ca_cal[:,2]
    pd.DataFrame(best_pred).to_csv(Path(OUT_DIR, "xdom_predictions.csv"), index=False)
    print(f"\nSaved predictions → {Path(OUT_DIR,'xdom_predictions.csv')}")

if __name__ == "__main__":
    main()
