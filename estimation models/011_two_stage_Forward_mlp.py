# ============================================================
# eit_two_stage_physics_final_ae_mlp.py  (Autoencoder-G + MLP core; BP-based σ)
# ------------------------------------------------------------
# - Auto-fit similarity: (u,v)_robot -> (x,y)_EIT (unit disk)
# - Stage A:
#     AE:       sigma_s <-> z             (unsupervised reconstruction)
#     R:        (x,y,F)_s -> z            (supervised to encoder targets)
#     F_hat:    sigma_s   -> v_s          (forward surrogate)
# - Stage B (UPDATED):
#     F (MLP):  v_s -> (x,y,F)_s, with physics consistency via:
#                y_pred -> R -> Dec -> σ_s -> F_hat -> v_s (match xb)
# - Reconstruction of σ matches real-time script: BP(normalize=True)
# - Metrics reported in ROBOT units (invert similarity).
# ============================================================

import os
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ---------------------------
# PyEIT (BP inverse)
# ---------------------------
from pyeit.mesh import create
from pyeit.eit.protocol import create as create_protocol
from pyeit.eit.fem import EITForward
from pyeit.eit.bp import BP   # use BP like real-time

# ===========================
# Config
# ===========================
DATA_PATH = "data/merged_with_probe.csv"
OUT_DIR   = "outputs_two_stage_real"
os.makedirs(OUT_DIR, exist_ok=True)

USE_GPU = torch.cuda.is_available()
device = torch.device("cuda" if USE_GPU else "cpu")
print("Using device:", device)

# Sensor / training knobs
SENSOR_RADIUS_M = 0.09
LAMBDA_PHYS = 0.01          # physics-consistency weight
FORCE_W_SUP  = 5.0          # supervised force upweight

# AE / latent sizes
Z_DIM = 16

# EIT / mesh configuration
N_EL = 16
PROTO_DIST_EXC = 1
PROTO_STEP_MEAS = 1
PARSER_MEAS = "std"         # match real-time

# Calibration sampling
FORCE_MIN_FOR_CAL = 0.8
MAX_CAL_SAMPLES   = 200

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

# ===========================
# Helpers
# ===========================
def element_centroids(mesh_obj):
    nodes = mesh_obj.node[:, :2]
    elems = mesh_obj.element
    return nodes[elems].mean(axis=1).astype(np.float32)  # [n_elem, 2]

def recon_hotspot_angle(ds, centroids):
    w = np.maximum(ds, 0.0)
    sw = w.sum()
    if sw <= 1e-12: return None, 0.0
    xy = (centroids * w[:, None]).sum(axis=0) / (sw + 1e-12)
    th = float(np.arctan2(xy[1], xy[0]))
    energy = float((w**2).sum())
    return th, energy

def fit_similarity_2d(P_robot, Q_eit):
    P = np.asarray(P_robot, float); Q = np.asarray(Q_eit, float)
    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    scale = (S.sum()) / (Pc**2).sum()
    t = Q.mean(axis=0) - scale * (R @ P.mean(axis=0))
    return float(scale), R, t

def apply_similarity(P_robot, s, R, t):
    P = np.asarray(P_robot, float)
    return (s * (P @ R.T)) + t

def invert_similarity(Q_eit, s, R, t, mirror_y=False):
    Q = np.asarray(Q_eit, float)
    P = ((Q - t) / s) @ R
    if mirror_y: P[:, 1] *= -1.0
    return P

def rms(a):
    a = np.asarray(a, float)
    return float(np.sqrt((a**2).mean()))

# ===========================
# Load & basic filtering
# ===========================
df = pd.read_csv(DATA_PATH)

# EIT channels
eit_cols = [c for c in df.columns if c.startswith("eit_") and c.lower() != "t_eit"]
if len(eit_cols) == 0:
    raise RuntimeError("No EIT columns (eit_*) found.")
eps = 1e-9
eit_cols = [c for c in eit_cols if (df[c].nunique(dropna=False) > 1 and df[c].std(skipna=True) > eps)]

# Targets
need_cols = ["contact_u_m", "contact_v_m", "force_n"]
for c in need_cols:
    if c not in df.columns:
        raise RuntimeError(f"Missing column '{c}' in CSV.")
targets_df = df[need_cols].copy()

# Masks
mask_contact   = targets_df["force_n"].notna() & (targets_df["force_n"] >= 0.5)
mask_nocontact = targets_df["force_n"].notna() & (targets_df["force_n"] < 0.5)
if mask_nocontact.sum() == 0:
    raise RuntimeError("No 'no-contact' rows (force_n < 0.5) to build V_ref.")

# Arrays
V_all = df[eit_cols].to_numpy(dtype=np.float32)
Y_all_robot = targets_df.to_numpy(dtype=np.float32)  # [u_m, v_m, force]

# Subset for contact training
V = V_all[mask_contact]
Y_robot = Y_all_robot[mask_contact]            # [u,v,force]
V_ref = V_all[mask_nocontact].mean(axis=0, keepdims=True).astype(np.float32)

print(f"Total rows: {len(df)} | Contact rows: {len(V)} | EIT dim: {V.shape[1]}")

# ===========================
# Mesh + Protocol (BP like real-time)
# ===========================
mesh = create(n_el=N_EL, h0=0.1)
protocol = create_protocol(
    n_el=N_EL,
    dist_exc=PROTO_DIST_EXC,
    step_meas=PROTO_STEP_MEAS,
    parser_meas=PARSER_MEAS,
)
_ = EITForward(mesh, protocol)   # not used further

# --- BP inverse (same as real-time, but no display scaling) ---
bp_inv = BP(mesh, protocol)
bp_inv.setup(weight="none")

def recon_sigma_bp(v1, v0):
    # normalize=True, log_scale=False as in live code
    return bp_inv.solve(v1, v0, normalize=True, log_scale=False).astype(np.float32)

def bp_node_to_element(ds_node: np.ndarray, mesh_obj) -> np.ndarray:
    # convert node-wise field to element-wise by averaging vertices of each triangle
    return ds_node[mesh_obj.element].mean(axis=1).astype(np.float32)

# ===========================
# Reconstruct Δσ (BP) and hotspot angles
# ===========================
centroids = element_centroids(mesh)
sigma_list, angles, energies = [], [], []

for i in range(V.shape[0]):
    v1 = V[i].astype(np.float32)
    v0 = V_ref.squeeze().astype(np.float32)
    ds = recon_sigma_bp(v1, v0)
    if ds.shape[0] == mesh.node.shape[0] and ds.shape[0] != mesh.element.shape[0]:
        ds = bp_node_to_element(ds, mesh)
    elif ds.shape[0] != mesh.element.shape[0]:
        raise RuntimeError(f"Unexpected BP length: {ds.shape[0]} (nodes={mesh.node.shape[0]}, elems={mesh.element.shape[0]})")
    sigma_list.append(ds)
    th, en = recon_hotspot_angle(ds, centroids)
    angles.append(th); energies.append(en)

Sigma = np.stack(sigma_list, axis=0).astype(np.float32)  # [N, n_elements]
n_elements = Sigma.shape[1]
angles = np.array(angles, dtype=object)
energies = np.array(energies, dtype=float)
print(f"Sigma shape: {Sigma.shape} | valid angles: {(angles != None).sum()} / {len(angles)}")

# ===========================
# Auto-calibrate (robot -> EIT)
# ===========================
valid = np.array([a is not None for a in angles])
strong = Y_robot[:, 2] >= FORCE_MIN_FOR_CAL
cand = np.where(valid & strong)[0]
if cand.size < 3:
    cand = np.where(valid)[0]
    print(f"[CAL] Warning: few strong-force samples; using {cand.size} valid angles.")

K = min(MAX_CAL_SAMPLES, cand.size)
pick = cand[np.argsort(energies[cand])[::-1][:K]]

P_robot = Y_robot[pick, :2]
ang_pick = np.asarray([float(a) for a in angles[pick]], dtype=float)
Q_eit   = np.column_stack([np.cos(ang_pick), np.sin(ang_pick)])

s1, R1, t1 = fit_similarity_2d(P_robot, Q_eit)
err1 = rms(apply_similarity(P_robot, s1, R1, t1) - Q_eit)
P_mir = P_robot.copy(); P_mir[:,1] *= -1.0
s2, R2, t2 = fit_similarity_2d(P_mir, Q_eit)
err2 = rms(apply_similarity(P_mir, s2, R2, t2) - Q_eit)

use_mirror_cal = err2 < err1
s_cal, R_cal, t_cal = (s2, R2, t2) if use_mirror_cal else (s1, R1, t1)
print(f"[CAL] similarity fit | pairs={K} | mirror_y={use_mirror_cal} | RMS={min(err1,err2):.4f}")
np.savez(Path(OUT_DIR, "robot2eit_similarity.npz"),
         s=s_cal, R=R_cal, t=t_cal, use_mirror=use_mirror_cal)

uv = Y_robot[:, :2].copy()
if use_mirror_cal: uv[:,1] *= -1.0
uv_eit = apply_similarity(uv, s_cal, R_cal, t_cal)
Y_eit  = np.column_stack([uv_eit, Y_robot[:, 2]])

# Optional quick visual check
# plt.tripcolor(mesh.node[:,0], mesh.node[:,1], mesh.element, Sigma[100], shading="flat"); plt.show()

# ===========================
# Train/val split
# ===========================
V_tr, V_te, Y_tr_eit, Y_te_eit, Sigma_tr, Sigma_te, Y_tr_robot, Y_te_robot = train_test_split(
    V, Y_eit, Sigma, Y_robot, test_size=0.2, random_state=RANDOM_SEED
)

# ===========================
# Scalers
# ===========================
sc_y = StandardScaler()       # (x_eit, y_eit, force)
sc_v = StandardScaler()       # voltages
sc_sigma = StandardScaler()   # per-element Δσ (BP-normalized)

Y_tr_s = sc_y.fit_transform(Y_tr_eit).astype(np.float32)
Y_te_s = sc_y.transform(Y_te_eit).astype(np.float32)
V_tr_s = sc_v.fit_transform(V_tr).astype(np.float32)
V_te_s = sc_v.transform(V_te).astype(np.float32)
Sigma_tr_s = sc_sigma.fit_transform(Sigma_tr).astype(np.float32)
Sigma_te_s = sc_sigma.transform(Sigma_te).astype(np.float32)

# ===========================
# Tensors
# ===========================
Y_tr_s_t = torch.tensor(Y_tr_s, dtype=torch.float32, device=device)
Y_te_s_t = torch.tensor(Y_te_s, dtype=torch.float32, device=device)
V_tr_s_t = torch.tensor(V_tr_s, dtype=torch.float32, device=device)   # [N, L]
V_te_s_t = torch.tensor(V_te_s, dtype=torch.float32, device=device)   # [N, L]
Sigma_tr_s_t = torch.tensor(Sigma_tr_s, dtype=torch.float32, device=device)
Sigma_te_s_t = torch.tensor(Sigma_te_s, dtype=torch.float32, device=device)

# ===========================
# Models
# ===========================
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=[256,128], out_dim=1, act=nn.ReLU, dropout=0.0, bn=False):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h)]
            if bn: layers += [nn.BatchNorm1d(h)]
            layers += [act()]
            if dropout > 0: layers += [nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# --- Autoencoder for sigma_s --------------------------------
class SigmaEncoder(nn.Module):
    def __init__(self, in_dim, z_dim=Z_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, z_dim),
            nn.LayerNorm(z_dim)         # stabilize latent scale
        )
    def forward(self, x): return self.net(x)

class SigmaDecoder(nn.Module):
    def __init__(self, z_dim=Z_DIM, out_dim=None):
        super().__init__()
        assert out_dim is not None
        self.net = nn.Sequential(
            nn.Linear(z_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, out_dim)
        )
    def forward(self, z): return self.net(z)

# ===========================
# Stage A1: Train AE on sigma_s
# ===========================
enc = SigmaEncoder(in_dim=Sigma_tr_s.shape[1], z_dim=Z_DIM).to(device)
dec = SigmaDecoder(z_dim=Z_DIM, out_dim=Sigma_tr_s.shape[1]).to(device)
opt_ae = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3, weight_decay=1e-4)
mse = nn.MSELoss()

print("\n=== Stage A1: train Sigma Autoencoder (σ_s <-> z) ===")
EPOCHS_AE = 300
BATCH = 128
Ntr = Sigma_tr_s_t.size(0)
best_rec = float("inf")

for ep in range(EPOCHS_AE):
    enc.train(); dec.train()
    perm = torch.randperm(Ntr, device=device)
    ep_loss = 0.0
    for i in range(0, Ntr, BATCH):
        idx = perm[i:i+BATCH]
        xb = Sigma_tr_s_t[idx]                       # σ_s
        z  = enc(xb)
        xb_hat = dec(z)                              # σ_s_hat
        loss = mse(xb_hat, xb)
        opt_ae.zero_grad(); loss.backward(); opt_ae.step()
        ep_loss += float(loss.item()) * xb.size(0)
    ep_loss /= Ntr
    if (ep % 25) == 0 or ep == EPOCHS_AE-1:
        enc.eval(); dec.eval()
        with torch.no_grad():
            z_te  = enc(Sigma_te_s_t)
            te_hat = dec(z_te)
            val = mse(te_hat, Sigma_te_s_t).item()
        print(f"[AE] Epoch {ep:03d} | train {ep_loss:.6f} | val_rec {val:.6f}")
        best_rec = min(best_rec, val)

# Freeze decoder for later physics
for p in dec.parameters(): p.requires_grad = False
dec.eval()

# ===========================
# Stage A2: Train ZReg: (x,y,F)_s -> z  (teacher = encoder)
# ===========================
with torch.no_grad():
    Z_tr_t = enc(Sigma_tr_s_t)    # [Ntr, Z_DIM]
    Z_te_t = enc(Sigma_te_s_t)    # [Nte, Z_DIM]

ZReg = MLP(in_dim=3, hidden=[256,128], out_dim=Z_DIM, dropout=0.0, bn=False).to(device)
opt_ZReg = optim.Adam(ZReg.parameters(), lr=1e-3, weight_decay=1e-4)

print("\n=== Stage A2: train R ((x,y,F)_s -> z) ===")
EPOCHS_R = 300
best_R = float("inf")
NtrY = Y_tr_s_t.size(0)

for ep in range(EPOCHS_R):
    ZReg.train()
    perm = torch.randperm(NtrY, device=device)
    ep_loss = 0.0
    for i in range(0, NtrY, 128):
        idx = perm[i:i+128]
        yb = Y_tr_s_t[idx]              # (x,y,F)_s
        zb = Z_tr_t[idx]                # teacher z
        zhat = ZReg(yb)
        loss = mse(zhat, zb)
        opt_ZReg.zero_grad(); loss.backward(); opt_ZReg.step()
        ep_loss += float(loss.item()) * yb.size(0)
    ep_loss /= NtrY
    if (ep % 25) == 0 or ep == EPOCHS_R-1:
        ZReg.eval()
        with torch.no_grad():
            val = mse(ZReg(Y_te_s_t), Z_te_t).item()
        print(f"[R] Epoch {ep:03d} | train {ep_loss:.6f} | val {val:.6f}")
        best_R = min(best_R, val)

# Freeze ZReg for Stage B (can fine-tune later if desired)
for p in ZReg.parameters(): p.requires_grad = False
ZReg.eval()

# ===========================
# Stage A3: Train F_hat: σ_s -> v_s
# ===========================
F_hat = MLP(in_dim=Sigma_tr_s.shape[1], hidden=[256,128], out_dim=V.shape[1], dropout=0.0, bn=False).to(device)

def train_simple(model, Xtr, Ytr, Xte, Yte, lr=1e-3, epochs=300, batch=128, tag="model"):
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    N = Xtr.size(0)
    best = float("inf")
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(N, device=device)
        ep_loss = 0.0
        for i in range(0, N, batch):
            idx = perm[i:i+batch]
            xb, yb = Xtr[idx], Ytr[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item()) * xb.size(0)
        ep_loss /= N
        if (ep % 25) == 0 or ep == epochs-1:
            model.eval()
            with torch.no_grad():
                val = loss_fn(model(Xte), Yte).item()
            print(f"[{tag}] Epoch {ep:03d} | train {ep_loss:.6f} | val {val:.6f}")
            best = min(best, val)
    return model

print("\n=== Stage A3: train F_hat (σ_s -> v_s) ===")
F_hat = train_simple(F_hat, Sigma_tr_s_t, V_tr_s_t, Sigma_te_s_t, V_te_s_t,
                     lr=1e-3, epochs=300, batch=128, tag="F_hat")

# Freeze F_hat
for p in F_hat.parameters(): p.requires_grad = False
F_hat.eval()

# Save Stage-A artifacts
torch.save(
    {"enc_state_dict": enc.state_dict(),
     "dec_state_dict": dec.state_dict(),
     "R_state_dict": ZReg.state_dict(),
     "Fhat_state_dict": F_hat.state_dict(),
     "sc_y_mean": sc_y.mean_, "sc_y_scale": sc_y.scale_,
     "sc_v_mean": sc_v.mean_, "sc_v_scale": sc_v.scale_,
     "sc_sigma_mean": sc_sigma.mean_, "sc_sigma_scale": sc_sigma.scale_},
    Path(OUT_DIR, "stageA_ae_R_Fhat.pt")
)

# ===========================
# Stage B: voltage -> (x,y,force) with physics loss (MLP CORE)
# ===========================
# Use plain [N, L] tensors for MLP
X_F_tr = V_tr_s_t            # [N, L]  scaled voltages
Y_F_tr = Y_tr_s_t            # [N, 3]  scaled (x,y,F)
X_F_te = V_te_s_t
Y_F_te = Y_te_s_t

class VoltageMLP(nn.Module):
    """MLP core: v_s (L,) -> (x,y,F)_s (3,)"""
    def __init__(self, in_dim, out_dim=3, hidden=[512, 256, 128], dropout=0.1, bn=True):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h)]
            if bn: layers += [nn.BatchNorm1d(h)]
            layers += [nn.ReLU()]
            if dropout > 0: layers += [nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

F = VoltageMLP(in_dim=V.shape[1], out_dim=3, hidden=[512,256,128], dropout=0.1, bn=True).to(device)
opt_F = optim.Adam(F.parameters(), lr=1e-3, weight_decay=1e-4)
mse = nn.MSELoss()

EPOCHS_B = 500
BATCH_B = 64
NtrB = X_F_tr.size(0)
best_val = float("inf")

print("\n=== Stage B: train F (MLP) with physics consistency ===")
for ep in range(EPOCHS_B):
    F.train()
    perm = torch.randperm(NtrB, device=device)
    ep_loss = 0.0
    for i in range(0, NtrB, BATCH_B):
        idx = perm[i:i+BATCH_B]
        xb = X_F_tr[idx]        # [B, L] v_s
        yb = Y_F_tr[idx]        # [B, 3] (x,y,F)_s

        opt_F.zero_grad()
        y_pred = F(xb)          # [B, 3] (x,y,F)_s

        # supervised (force upweighted)
        loss_sup = mse(y_pred[:, :2], yb[:, :2]) + FORCE_W_SUP * mse(y_pred[:, 2:], yb[:, 2:])

        # physics path (keep graph alive so grad flows to y_pred)
        z_pred   = ZReg(y_pred)       # z
        sigma_s  = dec(z_pred)        # σ_s
        v_sim_s  = F_hat(sigma_s)     # v̂_s
        v_true_s = xb                 # v_s
        loss_phys = mse(v_sim_s, v_true_s)

        loss = loss_sup + LAMBDA_PHYS * loss_phys
        loss.backward(); opt_F.step()
        ep_loss += float(loss.item()) * xb.size(0)
    ep_loss /= NtrB

    # validation
    if (ep % 25) == 0 or ep == EPOCHS_B - 1:
        F.eval()
        with torch.no_grad():
            y_val   = F(X_F_te)
            sup_val = (mse(y_val[:, :2], Y_F_te[:, :2]) +
                       FORCE_W_SUP * mse(y_val[:, 2:], Y_F_te[:, 2:])).item()

            z_val   = ZReg(y_val)
            sigma_v = dec(z_val)
            phys_v  = mse(F_hat(sigma_v), X_F_te).item()
            val_tot = sup_val + LAMBDA_PHYS * phys_v

        print(f"[StageB] Ep {ep:03d} | train {ep_loss:.6f} | val_sup {sup_val:.6f} | val_phys {phys_v:.6f} | val_total {val_tot:.6f}")

        if val_tot < best_val:
            best_val = val_tot
            torch.save({"F_state_dict": F.state_dict()}, Path(OUT_DIR, "F_stageB_best.pt"))
            print(f"  ↳ ✅ new best total={best_val:.6f}; saved to {OUT_DIR}/F_stageB_best.pt")

# ===========================
# Evaluation in ROBOT units (invert similarity)
# ===========================
F.eval()
with torch.no_grad():
    y_pred_te_s = F(X_F_te).cpu().numpy()
y_pred_te_eit = (y_pred_te_s * sc_y.scale_) + sc_y.mean_    # (x_eit, y_eit, force)

Q_eit = y_pred_te_eit[:, :2]
P_robot_pred = invert_similarity(Q_eit, s_cal, R_cal, t_cal, mirror_y=use_mirror_cal)
y_pred_robot = np.column_stack([P_robot_pred, y_pred_te_eit[:, 2]])
y_true_robot = Y_te_robot

mae = mean_absolute_error(y_true_robot, y_pred_robot, multioutput="raw_values")
r2  = r2_score(y_true_robot, y_pred_robot, multioutput="raw_values")

print("\n=== Final performance (ROBOT units) ===")
for name, m, r in zip(["u (m)", "v (m)", "force (N)"], mae, r2):
    print(f"MAE {name:7s}: {m:.4g} | R^2: {r:.4f}")

print(f"\n✅ Finished. BP-based reconstruction aligned with real-time; Stage-B uses MLP core. Artifacts in '{OUT_DIR}/'.")
