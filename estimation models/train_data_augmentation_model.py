# pretrain_transfer_plus.py
import os, json, copy, math, warnings
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# =========================
# Config
# =========================
SIM_DATA_PATH  = "data/touch_force_dataset.csv"       # <-- change if needed
REAL_DATA_PATH = "data/merged_with_probe.csv"         # <-- change if needed
OUT_DIR = "out_pretrain_transfer_plus"
os.makedirs(OUT_DIR, exist_ok=True)

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

USE_GPU = torch.cuda.is_available()
device = torch.device("cuda" if USE_GPU else "cpu")
print("Using device:", device)

# Splits
VAL_FRACTION  = 0.15
TEST_FRACTION = 0.20

# Training knobs
EPOCHS_PRETRAIN = 300
EPOCHS_FT_HEAD  = 80     # warm-up: train adapter+head with trunk frozen
EPOCHS_FT_FULL  = 220    # then unfreeze trunk
EPOCHS_SCRATCH  = 300

BATCH_SIZE      = 128
LR_PRETRAIN     = 1e-3
LR_FT_HEAD      = 1e-3
LR_FT_FULL      = 1e-4
LR_SCRATCH      = 1e-3
WEIGHT_DECAY    = 1e-4
FORCE_W         = 3.0      # upweight force loss
EARLY_STOP_PAT  = 30       # patience (epochs) based on val force R2

# L2-SP (fine-tune regularization toward pretrained trunk weights)
L2SP_ALPHA      = 1e-3     # try 1e-4 to 1e-2

# Domain randomization for sim (applied on-the-fly per batch)
SIM_AUG = dict(
    sigma_baseline=0.01,    # per-sample baseline shift
    sigma_gain=0.02,        # per-channel gain jitter
    p_drop=0.02,            # random channel drop prob
    block_size=None,        # if known, correlated drift per block (set to int)
    sigma_block=0.01
)

# Optional: if your real CSV has a "session" or "date" column, set this to use group-wise splits
REAL_SESSION_COL = None   # e.g., "session_id" or "date". If None, random splits are used.

# =========================
# Utils
# =========================
def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def evaluate_regression(y_true, y_pred, names=("x","y","force")) -> Dict:
    mae = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    r2  = r2_score(y_true, y_pred, multioutput="raw_values")
    return {k: {"MAE": float(m), "R2": float(r)} for k, m, r in zip(names, mae, r2)}

def force_r2(y_true, y_pred) -> float:
    # assumes last column is force
    return float(r2_score(y_true[:, 2], y_pred[:, 2]))

def split_train_val(X, y, val_fraction=0.15, rand=RANDOM_SEED, groups=None):
    if groups is None:
        return train_test_split(X, y, test_size=val_fraction, random_state=rand)
    gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=rand)
    idx_tr, idx_val = next(gss.split(X, y, groups))
    return X[idx_tr], X[idx_val], y[idx_tr], y[idx_val]

def sim_to_real_augment(Xb: np.ndarray, block_size=None, sigma_baseline=0.01,
                        sigma_gain=0.02, p_drop=0.02, sigma_block=0.01) -> np.ndarray:
    X = Xb.copy()
    B, D = X.shape
    # per-sample baseline shift (correlated across channels)
    shift = np.random.normal(0, sigma_baseline, size=(B, 1)).astype(np.float32)
    X += shift
    # per-channel gain jitter
    gains = np.random.normal(1.0, sigma_gain, size=(1, D)).astype(np.float32)
    X *= gains
    # correlated drift per block (if known)
    if block_size:
        for start in range(0, D, block_size):
            drift = np.random.normal(0, sigma_block, size=(B, 1)).astype(np.float32)
            X[:, start:start+block_size] += drift
    # random channel drop
    if p_drop > 0:
        mask = (np.random.rand(B, D) > p_drop).astype(np.float32)
        X *= mask
    return X

# =========================
# Data loaders
# =========================
def load_sim_csv(path) -> Tuple[np.ndarray,np.ndarray,Tuple[str,...]]:
    """
    Assumes sim CSV columns: [features..., x, y, force]
    """
    df = pd.read_csv(path)
    if df.shape[1] < 4:
        raise RuntimeError("Sim CSV must have features + 3 label columns.")
    X = df.iloc[:, :-3].to_numpy(np.float32)
    y = df.iloc[:, -3:].to_numpy(np.float32)
    return X, y, ("x","y","force")

def load_real_csv(path, force_min=0.5, session_col: Optional[str]=None):
    """
    Features: EIT columns start with 'eit_' (exclude 't_eit')
    Labels:   contact_u_m, contact_v_m, force_n
    Drop rows with force < force_min and NaNs
    Remove constant EIT channels
    Returns (X, y, label_names, eit_cols, groups)
    """
    df = pd.read_csv(path)
    need_cols = ["contact_u_m","contact_v_m","force_n"]
    for c in need_cols:
        if c not in df.columns:
            raise RuntimeError(f"Missing column '{c}' in real CSV.")

    eit_cols = [c for c in df.columns if c.startswith("eit_") and c.lower() != "t_eit"]
    if not eit_cols:
        raise RuntimeError("No EIT columns (eit_*) found in real CSV.")

    # remove near-constant channels
    eps = 1e-9
    eit_cols = [c for c in eit_cols if df[c].nunique(dropna=False) > 1 and df[c].std(skipna=True) > eps]
    if not eit_cols:
        raise RuntimeError("All EIT cols are constant/near-constant after filtering.")

    y_df = df[need_cols].copy()
    keep = y_df["force_n"].notna() & (y_df["force_n"] >= force_min)
    X = df.loc[keep, eit_cols].to_numpy(np.float32)
    y = y_df.loc[keep, ["contact_u_m","contact_v_m","force_n"]].to_numpy(np.float32)

    groups = None
    if session_col is not None and session_col in df.columns:
        groups = df.loc[keep, session_col].to_numpy()
    return X, y, ("x","y","force"), eit_cols, groups

# =========================
# Model
# =========================
class Trunk(nn.Module):
    def __init__(self, in_dim, hidden=(512, 256, 128), act=nn.ReLU):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), act()]
            d = h
        self.net = nn.Sequential(*layers)
        self.out_dim = d
    def forward(self, x): return self.net(x)

class Head(nn.Module):
    def __init__(self, in_dim, out_dim=3):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
    def forward(self, x): return self.fc(x)

class Adapter(nn.Module):
    """Small adapter to map real features to sim-trained feature space."""
    def __init__(self, in_dim, hidden=0):
        super().__init__()
        if hidden and hidden > 0:
            self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, in_dim))
        else:
            self.net = nn.Linear(in_dim, in_dim)  # affine
            nn.init.eye_(self.net.weight) if hasattr(nn.init, "eye_") and self.net.weight.shape[0]==self.net.weight.shape[1] else None
            nn.init.zeros_(self.net.bias)
    def forward(self, x): return self.net(x)

class ModelWithAdapter(nn.Module):
    def __init__(self, in_dim, trunk_hidden=(512,256,128), out_dim=3, adapter_hidden=0):
        super().__init__()
        self.adapter = Adapter(in_dim, hidden=adapter_hidden)
        self.trunk = Trunk(in_dim, hidden=trunk_hidden)
        self.head   = Head(self.trunk.out_dim, out_dim=out_dim)
    def forward(self, x):
        x = self.adapter(x)
        feat = self.trunk(x)
        return self.head(feat)

class SimpleModel(nn.Module):
    """No adapter; plain trunk+head."""
    def __init__(self, in_dim, trunk_hidden=(512,256,128), out_dim=3):
        super().__init__()
        self.trunk = Trunk(in_dim, hidden=trunk_hidden)
        self.head  = Head(self.trunk.out_dim, out_dim=out_dim)
    def forward(self, x):
        return self.head(self.trunk(x))

# =========================
# Training helpers
# =========================
def batch_iter(N, batch_size, device):
    perm = torch.randperm(N, device=device)
    for i in range(0, N, batch_size):
        yield perm[i:i+batch_size]

def sup_loss(pred, target, force_w=1.0, mse=nn.MSELoss()):
    return mse(pred[:, :2], target[:, :2]) + force_w * mse(pred[:, 2:], target[:, 2:])

def fit_loop(model, Xtr, ytr, Xval, yval, epochs, lr, weight_decay, force_w,
             prefix="model", early_stop_pat=30,
             sim_augment: Optional[dict]=None,
             freeze_parts: Optional[List[str]]=None,
             l2sp_alpha: float=0.0,
             l2sp_refs: Optional[Dict[str, torch.Tensor]]=None):
    """
    Training loop with early stop by val force R2.
    freeze_parts: list of module name prefixes to freeze (e.g., ["trunk."])
    l2sp_refs: dict of param_name -> tensor (pretrained refs), applied to params that require_grad.
    """
    model = model.to(device)
    model.train()

    # Freeze requested parts
    if freeze_parts:
        for name, p in model.named_parameters():
            if any(name.startswith(pref) for pref in freeze_parts):
                p.requires_grad = False

    # optimizer on trainable params only
    params = [p for p in model.parameters() if p.requires_grad]
    opt = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    mse = nn.MSELoss()

    # tensors
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    Xval_t = torch.tensor(Xval, dtype=torch.float32, device=device)
    yval_t = torch.tensor(yval, dtype=torch.float32, device=device)

    N = Xtr.shape[0]
    best_force_r2 = -1e9
    best_state = copy.deepcopy(model.state_dict())
    patience = early_stop_pat
    history = []

    for ep in range(epochs):
        model.train()
        ep_loss = 0.0
        for idx in batch_iter(N, BATCH_SIZE, device):
            xb = Xtr_t[idx]
            yb = ytr_t[idx]

            # optional sim augment (expects numpy, so roundtrip)
            if sim_augment is not None:
                x_np = xb.detach().cpu().numpy()
                x_np = sim_to_real_augment(
                    x_np,
                    block_size=sim_augment.get("block_size"),
                    sigma_baseline=sim_augment.get("sigma_baseline", 0.01),
                    sigma_gain=sim_augment.get("sigma_gain", 0.02),
                    p_drop=sim_augment.get("p_drop", 0.02),
                    sigma_block=sim_augment.get("sigma_block", 0.01),
                )
                xb = torch.tensor(x_np, dtype=torch.float32, device=device)

            opt.zero_grad()
            pred = model(xb)
            loss = sup_loss(pred, yb, force_w=force_w, mse=mse)

            # L2-SP regularization toward pretrained refs
            if l2sp_alpha > 0.0 and l2sp_refs:
                l2sp = 0.0
                for name, p in model.named_parameters():
                    if p.requires_grad and name in l2sp_refs:
                        ref = l2sp_refs[name].to(p.device)
                        l2sp = l2sp + torch.sum((p - ref)**2)
                loss = loss + l2sp_alpha * l2sp

            loss.backward()
            opt.step()
            ep_loss += float(loss.item()) * xb.size(0)
        ep_loss /= N

        # validation (force R2 for early stop)
        model.eval()
        with torch.no_grad():
            pv = model(Xval_t).cpu().numpy()
            yv = yval_t.cpu().numpy()
            fr2 = force_r2(yv, pv)
            val_loss = float(sup_loss(torch.tensor(pv), torch.tensor(yv), force_w=force_w).item())

        history.append({"epoch": ep, "train_loss": ep_loss, "val_force_r2": fr2, "val_loss": val_loss})
        if (ep % 25) == 0 or ep == epochs-1:
            print(f"[{prefix}] Ep {ep:03d} | train {ep_loss:.6f} | val_loss {val_loss:.6f} | val_force_R2 {fr2:.4f}")

        improved = fr2 > best_force_r2 + 1e-6
        if improved:
            best_force_r2 = fr2
            best_state = copy.deepcopy(model.state_dict())
            patience = early_stop_pat
        else:
            patience -= 1
            if patience <= 0:
                print(f"[{prefix}] Early stopping at epoch {ep} (best val force R2={best_force_r2:.4f})")
                break

    model.load_state_dict(best_state)
    return model, history, best_force_r2

def eval_model(model, Xte, yte, scy=None, names=("x","y","force")):
    model.eval()
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred_s = model(Xte_t).cpu().numpy()
    yte_s = yte
    if scy is not None:
        pred = pred_s * scy.scale_ + scy.mean_
        ytrue = yte_s * scy.scale_ + scy.mean_
    else:
        pred = pred_s; ytrue = yte_s
    metrics = evaluate_regression(ytrue, pred, names)
    return metrics, pred, ytrue

def reinit_head(model):
    if hasattr(model, "head") and isinstance(model.head, Head):
        for m in model.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

# =========================
# Pipeline
# =========================
def run():
    # 1) ------------------ PRETRAIN ON SIM ------------------
    print("\n=== Loading SIM data (pretraining) ===")
    X_sim, y_sim, lbls = load_sim_csv(SIM_DATA_PATH)
    X_sim_tr, X_sim_te, y_sim_tr, y_sim_te = train_test_split(
        X_sim, y_sim, test_size=TEST_FRACTION, random_state=RANDOM_SEED
    )
    scx_sim = StandardScaler().fit(X_sim_tr)
    scy_sim = StandardScaler().fit(y_sim_tr)
    X_sim_tr_s = scx_sim.transform(X_sim_tr).astype(np.float32)
    X_sim_te_s = scx_sim.transform(X_sim_te).astype(np.float32)
    y_sim_tr_s = scy_sim.transform(y_sim_tr).astype(np.float32)
    y_sim_te_s = scy_sim.transform(y_sim_te).astype(np.float32)

    X_sim_tr_s, X_sim_val_s, y_sim_tr_s, y_sim_val_s = split_train_val(
        X_sim_tr_s, y_sim_tr_s, VAL_FRACTION, RANDOM_SEED
    )

    model_sim = SimpleModel(in_dim=X_sim_tr_s.shape[1], trunk_hidden=(512,256,128), out_dim=3).to(device)
    model_sim, hist_sim, best_fr2_sim = fit_loop(
        model_sim, X_sim_tr_s, y_sim_tr_s, X_sim_val_s, y_sim_val_s,
        epochs=EPOCHS_PRETRAIN, lr=LR_PRETRAIN, weight_decay=WEIGHT_DECAY,
        force_w=FORCE_W, prefix="Pretrain(SIM)", early_stop_pat=EARLY_STOP_PAT,
        sim_augment=SIM_AUG, freeze_parts=None, l2sp_alpha=0.0
    )
    m_sim_te, _, _ = eval_model(model_sim, X_sim_te_s, y_sim_te_s, scy_sim, names=lbls)
    print("\n[Pretrain(SIM)] Test metrics:", m_sim_te)

    ckpt_sim = {
        "trunk": model_sim.trunk.state_dict(),
        "head":  model_sim.head.state_dict(),
        "x_scaler_mean": scx_sim.mean_, "x_scaler_scale": scx_sim.scale_,
        "y_scaler_mean": scy_sim.mean_, "y_scaler_scale": scy_sim.scale_,
        "hist": hist_sim, "best_val_force_R2": best_fr2_sim,
    }
    torch.save(ckpt_sim, Path(OUT_DIR, "pretrained_on_sim.pth"))
    save_json(m_sim_te, Path(OUT_DIR, "pretrained_on_sim_metrics.json"))

    # Build L2-SP reference dict (only for trunk params)
    l2sp_refs = {f"trunk.{k}": v.clone().detach() for k, v in model_sim.trunk.state_dict().items()}

    # 2) ------------------ LOAD REAL DATA -------------------
    print("\n=== Loading REAL data ===")
    X_real, y_real, lbls, eit_cols, groups = load_real_csv(REAL_DATA_PATH, force_min=0.5, session_col=REAL_SESSION_COL)

    # hold-out test split on REAL (group-wise if groups provided)
    if groups is None:
        X_r_tr, X_r_te, y_r_tr, y_r_te = train_test_split(
            X_real, y_real, test_size=TEST_FRACTION, random_state=RANDOM_SEED
        )
        groups_tr = None
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=TEST_FRACTION, random_state=RANDOM_SEED)
        idx_tr, idx_te = next(gss.split(X_real, y_real, groups))
        X_r_tr, X_r_te, y_r_tr, y_r_te = X_real[idx_tr], X_real[idx_te], y_real[idx_tr], y_real[idx_te]
        groups_tr = groups[idx_tr]

    # Scalers fit on REAL-TRAIN ONLY (used for both fine-tune and scratch)
    scx_real = StandardScaler().fit(X_r_tr)
    scy_real = StandardScaler().fit(y_r_tr)

    X_r_tr_s = scx_real.transform(X_r_tr).astype(np.float32)
    X_r_te_s = scx_real.transform(X_r_te).astype(np.float32)
    y_r_tr_s = scy_real.transform(y_r_tr).astype(np.float32)
    y_r_te_s = scy_real.transform(y_r_te).astype(np.float32)

    # Internal train/val on real-train
    X_r_tr_s, X_r_val_s, y_r_tr_s, y_r_val_s = split_train_val(
        X_r_tr_s, y_r_tr_s, VAL_FRACTION, RANDOM_SEED, groups=groups_tr
    )

    # 3) ------------------ FINETUNE: ADAPTER + HEAD first (trunk frozen) ----
    print("\n=== Fine-tune on REAL (adapter+head, trunk frozen) ===")
    model_ft = ModelWithAdapter(in_dim=X_r_tr_s.shape[1], trunk_hidden=(512,256,128), out_dim=3, adapter_hidden=0).to(device)

    # load trunk weights from sim (loose: only matching shapes)
    sd_trunk = torch.load(Path(OUT_DIR, "pretrained_on_sim.pth"), map_location="cpu")["trunk"]
    cur_trunk = model_ft.trunk.state_dict()
    for k, v in sd_trunk.items():
        if k in cur_trunk and cur_trunk[k].shape == v.shape:
            cur_trunk[k] = v
    model_ft.trunk.load_state_dict(cur_trunk)

    # re-init head & adapter (fresh for real)
    reinit_head(model_ft)
    for m in model_ft.adapter.modules():
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.bias)
            if m.weight.shape[0] == m.weight.shape[1]:
                with torch.no_grad():
                    m.weight.copy_(torch.eye(m.weight.shape[0]))

    # freeze trunk; train adapter+head
    model_ft, hist_head, best_fr2_head = fit_loop(
        model_ft, X_r_tr_s, y_r_tr_s, X_r_val_s, y_r_val_s,
        epochs=EPOCHS_FT_HEAD, lr=LR_FT_HEAD, weight_decay=WEIGHT_DECAY,
        force_w=FORCE_W, prefix="Finetune(REAL)-HeadOnly", early_stop_pat=EARLY_STOP_PAT,
        freeze_parts=["trunk."], l2sp_alpha=0.0, l2sp_refs=None
    )

    # 4) ------------------ FINETUNE: UNFREEZE TRUNK (small LR + L2-SP) -----
    print("\n=== Fine-tune on REAL (unfreeze trunk, L2-SP) ===")
    # Unfreeze all
    for p in model_ft.parameters():
        p.requires_grad = True

    # Build param refs for L2-SP (match by full name)
    pretrained_refs = {}
    for name, p in model_ft.named_parameters():
        if name.startswith("trunk."):
            # map to trunk param key (remove 'trunk.' prefix) in sim refs, if exists
            key = name[len("trunk."):]
            if key in l2sp_refs:
                pretrained_refs[name] = l2sp_refs[key].clone().detach()

    model_ft, hist_full, best_fr2_full = fit_loop(
        model_ft, X_r_tr_s, y_r_tr_s, X_r_val_s, y_r_val_s,
        epochs=EPOCHS_FT_FULL, lr=LR_FT_FULL, weight_decay=WEIGHT_DECAY,
        force_w=FORCE_W, prefix="Finetune(REAL)-Full", early_stop_pat=EARLY_STOP_PAT,
        freeze_parts=None, l2sp_alpha=L2SP_ALPHA, l2sp_refs=pretrained_refs
    )

    # Evaluate on REAL-TEST
    m_ft_te, _, _ = eval_model(model_ft, X_r_te_s, y_r_te_s, scy_real, names=lbls)
    print("\n[Finetune(REAL)] Test metrics:", m_ft_te)

    torch.save({
        "adapter": model_ft.adapter.state_dict(),
        "trunk":   model_ft.trunk.state_dict(),
        "head":    model_ft.head.state_dict(),
        "x_scaler_mean": scx_real.mean_, "x_scaler_scale": scx_real.scale_,
        "y_scaler_mean": scy_real.mean_, "y_scaler_scale": scy_real.scale_,
        "hist_head": hist_head, "hist_full": hist_full
    }, Path(OUT_DIR, "finetuned_on_real_from_sim_plus.pth"))
    save_json(m_ft_te, Path(OUT_DIR, "finetuned_on_real_plus_metrics.json"))

    # 5) ------------------ SCRATCH ON REAL (baseline) ----------------------
    print("\n=== Train from scratch on REAL (no pretrain) ===")
    model_sc = SimpleModel(in_dim=X_r_tr_s.shape[1], trunk_hidden=(512,256,128), out_dim=3).to(device)
    # head+trunk both trainable
    model_sc, hist_sc, best_fr2_sc = fit_loop(
        model_sc, X_r_tr_s, y_r_tr_s, X_r_val_s, y_r_val_s,
        epochs=EPOCHS_SCRATCH, lr=LR_SCRATCH, weight_decay=WEIGHT_DECAY,
        force_w=FORCE_W, prefix="Scratch(REAL)", early_stop_pat=EARLY_STOP_PAT,
        freeze_parts=None, l2sp_alpha=0.0, l2sp_refs=None
    )
    m_sc_te, _, _ = eval_model(model_sc, X_r_te_s, y_r_te_s, scy_real, names=lbls)
    print("\n[Scratch(REAL)] Test metrics:", m_sc_te)

    torch.save({
        "trunk":   model_sc.trunk.state_dict(),
        "head":    model_sc.head.state_dict(),
        "x_scaler_mean": scx_real.mean_, "x_scaler_scale": scx_real.scale_,
        "y_scaler_mean": scy_real.mean_, "y_scaler_scale": scy_real.scale_,
        "hist": hist_sc
    }, Path(OUT_DIR, "scratch_on_real_plus.pth"))
    save_json(m_sc_te, Path(OUT_DIR, "scratch_on_real_plus_metrics.json"))

    # 6) ------------------ SUMMARY ----------------------------------------
    summary = {
        "Pretrain_SIM_test": m_sim_te,
        "Finetune_on_REAL_test": m_ft_te,
        "Scratch_on_REAL_test": m_sc_te
    }
    save_json(summary, Path(OUT_DIR, "summary_plus.json"))
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    run()
