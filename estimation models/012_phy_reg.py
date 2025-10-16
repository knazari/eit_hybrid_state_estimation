# ============================================================
# eit_latent_phy_regularizer_mlp.py
# ------------------------------------------------------------
# Same preprocessing as your working BP+AE script.
# New bits:
#   • Stage A3b: VoltEnc : v_s -> z    (teacher = enc(sigma_s))
#   • Stage B : F_mlp : v_s -> (x,y,F)_s
#          Loss = L_sup + λ_latent * || ZReg(y_pred) - VoltEnc(v_s) ||^2
# Evaluates in ROBOT units via your similarity fit.
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
from pyeit.mesh import create
from pyeit.eit.protocol import create as create_protocol
from pyeit.eit.fem import EITForward
from pyeit.eit.bp import BP


plt.rcParams['pdf.fonttype'] = 42   # use TrueType fonts, not Type 3
plt.rcParams['ps.fonttype'] = 42


# ===========================
# Config
# ===========================
DATA_PATH = "data/merged_with_probe.csv"
OUT_DIR   = "outputs_two_stage_real"
os.makedirs(OUT_DIR, exist_ok=True)

USE_GPU = torch.cuda.is_available()
device = torch.device("cuda" if USE_GPU else "cpu")
print("Using device:", device)

# Training knobs
SENSOR_RADIUS_M = 0.09
FORCE_W_SUP  = 5.0      # supervised force upweight
LAMBDA_LATENT = 0.03     # latent physics regularizer weight (try 0.1–0.5) -> 0.03 gave best force prediction among tested values

# Latent size
Z_DIM = 16 #-> 16 gave best force prediction for tested vlaues

# EIT / BP configuration (unchanged)
N_EL = 16
PROTO_DIST_EXC = 1
PROTO_STEP_MEAS = 1
PARSER_MEAS = "std"

# Calibration sampling (unchanged)
FORCE_MIN_FOR_CAL = 0.8
MAX_CAL_SAMPLES   = 200

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED); np.random.seed(RANDOM_SEED)

# ===========================
# Helpers (unchanged)
# ===========================
def element_centroids(mesh_obj):
    nodes = mesh_obj.node[:, :2]
    elems = mesh_obj.element
    return nodes[elems].mean(axis=1).astype(np.float32)

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
# Load & basic filtering (UNCHANGED)
# ===========================
df = pd.read_csv(DATA_PATH)

eit_cols = [c for c in df.columns if c.startswith("eit_") and c.lower() != "t_eit"]
if len(eit_cols) == 0:
    raise RuntimeError("No EIT columns (eit_*) found.")
eps = 1e-9
eit_cols = [c for c in eit_cols if (df[c].nunique(dropna=False) > 1 and df[c].std(skipna=True) > eps)]

need_cols = ["contact_u_m", "contact_v_m", "force_n"]
for c in need_cols:
    if c not in df.columns:
        raise RuntimeError(f"Missing column '{c}' in CSV.")
targets_df = df[need_cols].copy()

mask_contact   = targets_df["force_n"].notna() & (targets_df["force_n"] >= 0.5)
mask_nocontact = targets_df["force_n"].notna() & (targets_df["force_n"] < 0.5)
if mask_nocontact.sum() == 0:
    raise RuntimeError("No 'no-contact' rows (force_n < 0.5) to build V_ref.")

V_all = df[eit_cols].to_numpy(dtype=np.float32)
Y_all_robot = targets_df.to_numpy(dtype=np.float32)  # [u, v, force]

V = V_all[mask_contact]
Y_robot = Y_all_robot[mask_contact]
V_ref = V_all[mask_nocontact].mean(axis=0, keepdims=True).astype(np.float32)

print(f"Total rows: {len(df)} | Contact rows: {len(V)} | EIT dim: {V.shape[1]}")

# ===========================
# Mesh + BP inverse (UNCHANGED)
# ===========================
mesh = create(n_el=N_EL, h0=0.1)
protocol = create_protocol(
    n_el=N_EL, dist_exc=PROTO_DIST_EXC, step_meas=PROTO_STEP_MEAS, parser_meas=PARSER_MEAS
)
_ = EITForward(mesh, protocol)

bp_inv = BP(mesh, protocol)
bp_inv.setup(weight="none")

def recon_sigma_bp(v1, v0):
    return bp_inv.solve(v1, v0, normalize=True, log_scale=False).astype(np.float32)

def bp_node_to_element(ds_node, mesh_obj):
    return ds_node[mesh_obj.element].mean(axis=1).astype(np.float32)

# ===========================
# Reconstruct Δσ (BP) + angles (UNCHANGED)
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

Sigma = np.stack(sigma_list, axis=0).astype(np.float32)
n_elements = Sigma.shape[1]
angles = np.array(angles, dtype=object)
energies = np.array(energies, dtype=float)
print(f"Sigma shape: {Sigma.shape} | valid angles: {(angles != None).sum()} / {len(angles)}")

# ===========================
# Auto-calibrate (robot -> EIT) (UNCHANGED)
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

# Optional quick look (unchanged)
# plt.tripcolor(mesh.node[:,0], mesh.node[:,1], mesh.element, Sigma[100], shading="flat")
# plt.show()

# ===========================
# Train/val split (UNCHANGED)
# ===========================
V_tr, V_te, Y_tr_eit, Y_te_eit, Sigma_tr, Sigma_te, Y_tr_robot, Y_te_robot = train_test_split(
    V, Y_eit, Sigma, Y_robot, test_size=0.2, random_state=RANDOM_SEED
)

# ===========================
# Scalers (UNCHANGED)
# ===========================
sc_y = StandardScaler()
sc_v = StandardScaler()
sc_sigma = StandardScaler()

Y_tr_s = sc_y.fit_transform(Y_tr_eit).astype(np.float32)
Y_te_s = sc_y.transform(Y_te_eit).astype(np.float32)
V_tr_s = sc_v.fit_transform(V_tr).astype(np.float32)
V_te_s = sc_v.transform(V_te).astype(np.float32)
Sigma_tr_s = sc_sigma.fit_transform(Sigma_tr).astype(np.float32)
Sigma_te_s = sc_sigma.transform(Sigma_te).astype(np.float32)

# ===========================
# Tensors (UNCHANGED)
# ===========================
Y_tr_s_t = torch.tensor(Y_tr_s, dtype=torch.float32, device=device)
Y_te_s_t = torch.tensor(Y_te_s, dtype=torch.float32, device=device)
V_tr_s_t = torch.tensor(V_tr_s, dtype=torch.float32, device=device)
V_te_s_t = torch.tensor(V_te_s, dtype=torch.float32, device=device)
Sigma_tr_s_t = torch.tensor(Sigma_tr_s, dtype=torch.float32, device=device)
Sigma_te_s_t = torch.tensor(Sigma_te_s, dtype=torch.float32, device=device)

# ===========================
# Models
# ===========================
class MLP(nn.Module):
    def __init__(self, in_dim, hidden=[256,128], out_dim=1, act=nn.ReLU):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), act()]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# --- Autoencoder for sigma_s (UNCHANGED) ---
class SigmaEncoder(nn.Module):
    def __init__(self, in_dim, z_dim=Z_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, z_dim),
            nn.LayerNorm(z_dim)
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

# --- New: Voltage encoder to latent ---
class VoltEncoder(nn.Module):
    """v_s -> z, trained to match encoder(enc(sigma_s))"""
    def __init__(self, in_dim, z_dim=Z_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, z_dim)
        )
    def forward(self, v): return self.net(v)

mse = nn.MSELoss()

# ===========================
# Stage A1: AE on sigma_s (UNCHANGED)
# ===========================
enc = SigmaEncoder(in_dim=Sigma_tr_s.shape[1], z_dim=Z_DIM).to(device)
dec = SigmaDecoder(z_dim=Z_DIM, out_dim=Sigma_tr_s.shape[1]).to(device)
opt_ae = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3, weight_decay=1e-4)

print("\n=== Stage A1: train Sigma Autoencoder (σ_s <-> z) ===")
EPOCHS_AE = 300
BATCH = 128
Ntr = Sigma_tr_s_t.size(0)
best_rec = float("inf")

for ep in range(EPOCHS_AE):
    enc.train(); dec.train()
    perm = torch.randperm(Ntr, device=device); ep_loss = 0.0
    for i in range(0, Ntr, BATCH):
        idx = perm[i:i+BATCH]
        xb = Sigma_tr_s_t[idx]
        z  = enc(xb)
        xb_hat = dec(z)
        loss = mse(xb_hat, xb)
        opt_ae.zero_grad(); loss.backward(); opt_ae.step()
        ep_loss += float(loss.item()) * xb.size(0)
    ep_loss /= Ntr
    if (ep % 25) == 0 or ep == EPOCHS_AE-1:
        enc.eval(); dec.eval()
        with torch.no_grad():
            val = mse(dec(enc(Sigma_te_s_t)), Sigma_te_s_t).item()
        print(f"[AE] Epoch {ep:03d} | train {ep_loss:.6f} | val_rec {val:.6f}")
        best_rec = min(best_rec, val)

# Freeze decoder for later (we won’t need it in Stage B, but leaving as-is)
for p in dec.parameters(): p.requires_grad = False
dec.eval()

# ===========================
# Stage A2: ZReg: (x,y,F)_s -> z  (teacher = enc)
# ===========================
with torch.no_grad():
    Z_tr_t = enc(Sigma_tr_s_t)
    Z_te_t = enc(Sigma_te_s_t)

ZReg = MLP(in_dim=3, hidden=[256,128], out_dim=Z_DIM).to(device)
opt_ZReg = optim.Adam(ZReg.parameters(), lr=1e-3, weight_decay=1e-4)

print("\n=== Stage A2: train ZReg ((x,y,F)_s -> z) ===")
EPOCHS_R = 300
best_R = float("inf"); NtrY = Y_tr_s_t.size(0)

for ep in range(EPOCHS_R):
    ZReg.train()
    perm = torch.randperm(NtrY, device=device); ep_loss = 0.0
    for i in range(0, NtrY, BATCH):
        idx = perm[i:i+BATCH]
        yb = Y_tr_s_t[idx]; zb = Z_tr_t[idx]
        zhat = ZReg(yb)
        loss = mse(zhat, zb)
        opt_ZReg.zero_grad(); loss.backward(); opt_ZReg.step()
        ep_loss += float(loss.item()) * yb.size(0)
    ep_loss /= NtrY
    if (ep % 25) == 0 or ep == EPOCHS_R-1:
        ZReg.eval()
        with torch.no_grad():
            val = mse(ZReg(Y_te_s_t), Z_te_t).item()
        print(f"[ZReg] Epoch {ep:03d} | train {ep_loss:.6f} | val {val:.6f}")
        best_R = min(best_R, val)

for p in ZReg.parameters(): p.requires_grad = False
ZReg.eval()

# ===========================
# Stage A3b: VoltEnc: v_s -> z  (teacher = enc(sigma_s))
# ===========================
VoltEnc = VoltEncoder(in_dim=V_tr_s.shape[1], z_dim=Z_DIM).to(device)
opt_VE = optim.Adam(VoltEnc.parameters(), lr=1e-3, weight_decay=1e-4)

print("\n=== Stage A3b: train VoltEnc (v_s -> z) ===")
EPOCHS_VE = 300
best_VE = float("inf"); NtrV = V_tr_s_t.size(0)

for ep in range(EPOCHS_VE):
    VoltEnc.train()
    perm = torch.randperm(NtrV, device=device); ep_loss = 0.0
    for i in range(0, NtrV, BATCH):
        idx = perm[i:i+BATCH]
        vb = V_tr_s_t[idx]
        zb_teacher = Z_tr_t[idx]   # teacher from enc(sigma_s)
        zb_hat = VoltEnc(vb)
        loss = mse(zb_hat, zb_teacher)
        opt_VE.zero_grad(); loss.backward(); opt_VE.step()
        ep_loss += float(loss.item()) * vb.size(0)
    ep_loss /= NtrV
    if (ep % 25) == 0 or ep == EPOCHS_VE-1:
        VoltEnc.eval()
        with torch.no_grad():
            val = mse(VoltEnc(V_te_s_t), Z_te_t).item()
        print(f"[VoltEnc] Epoch {ep:03d} | train {ep_loss:.6f} | val {val:.6f}")
        best_VE = min(best_VE, val)

for p in VoltEnc.parameters(): p.requires_grad = False
VoltEnc.eval()

# ===========================
# Stage B: MLP v_s -> (x,y,F)_s with latent physics regularizer
# ===========================
F_mlp = MLP(in_dim=V.shape[1], hidden=[512,256,128], out_dim=3).to(device)
opt_F = optim.Adam(F_mlp.parameters(), lr=1e-3)
mse = nn.MSELoss()

EPOCHS_B = 500
BATCH_B = 64
NtrB = V_tr_s_t.size(0)
best_val = float("inf")


print("\n=== Stage B: train MLP (v_s -> (x,y,F)_s) with latent physics prior ===")
for ep in range(EPOCHS_B):
    F_mlp.train()
    perm = torch.randperm(NtrB, device=device); ep_loss = 0.0
    for i in range(0, NtrB, BATCH_B):
        idx = perm[i:i+BATCH_B]
        vb = V_tr_s_t[idx]        # [B, L] v_s
        yb = Y_tr_s_t[idx]        # [B, 3] (x,y,F)_s

        opt_F.zero_grad()
        y_pred = F_mlp(vb)

        # supervised (force upweighted)
        loss_sup = mse(y_pred[:, :2], yb[:, :2]) + FORCE_W_SUP * mse(y_pred[:, 2:], yb[:, 2:])

        # latent consistency: ZReg(y_pred) ≈ VoltEnc(v_true)
        z_from_y = ZReg(y_pred)
        z_from_v = VoltEnc(vb)
        loss_lat = mse(z_from_y, z_from_v)

        loss = loss_sup + LAMBDA_LATENT * loss_lat
        loss.backward(); opt_F.step()
        ep_loss += float(loss.item()) * vb.size(0)
    ep_loss /= NtrB

    # validation
    if (ep % 25) == 0 or ep == EPOCHS_B - 1:
        F_mlp.eval()
        with torch.no_grad():
            y_val   = F_mlp(V_te_s_t)
            sup_val = (mse(y_val[:, :2], Y_te_s_t[:, :2]) +
                       FORCE_W_SUP * mse(y_val[:, 2:], Y_te_s_t[:, 2:])).item()
            lat_v   = mse(ZReg(y_val), VoltEnc(V_te_s_t)).item()
            val_tot = sup_val + LAMBDA_LATENT * lat_v
        print(f"[StageB] Ep {ep:03d} | train {ep_loss:.6f} | val_sup {sup_val:.6f} | val_lat {lat_v:.6f} | val_total {val_tot:.6f}")

        if val_tot < best_val:
            best_val = val_tot
            torch.save({"F_state_dict": F_mlp.state_dict()}, Path(OUT_DIR, "F_stageB_best_mlp.pt"))
            print(f"  ↳ ✅ new best total={best_val:.6f}; saved to {OUT_DIR}/F_stageB_best_mlp.pt")


# ===========================
# Evaluation in ROBOT units (UNCHANGED)
# ===========================
F_mlp.eval()
with torch.no_grad():
    y_pred_te_s = F_mlp(V_te_s_t).cpu().numpy()
y_pred_te_eit = (y_pred_te_s * sc_y.scale_) + sc_y.mean_

Q_eit = y_pred_te_eit[:, :2]
P_robot_pred = invert_similarity(Q_eit, s_cal, R_cal, t_cal, mirror_y=use_mirror_cal)
y_pred_robot = np.column_stack([P_robot_pred, y_pred_te_eit[:, 2]])
y_true_robot = Y_te_robot

mae = mean_absolute_error(y_true_robot, y_pred_robot, multioutput="raw_values")
r2  = r2_score(y_true_robot, y_pred_robot, multioutput="raw_values")

print("\n=== Final performance (ROBOT units) ===")
for name, m, r in zip(["u (m)", "v (m)", "force (N)"], mae, r2):
    print(f"MAE {name:7s}: {m:.4g} | R^2: {r:.4f}")

print(f"\n✅ Finished. Artifacts in '{OUT_DIR}/'.")



def parity_plot(y_true, y_pred, title, xlabel, ylabel, out_path):
    lim_min = float(min(np.min(y_true), np.min(y_pred)))
    lim_max = float(max(np.max(y_true), np.max(y_pred)))
    pad = 0.05 * (lim_max - lim_min + 1e-12)
    lim_min -= pad; lim_max += pad

    plt.figure(figsize=(5.2, 5.2))
    plt.scatter(y_true, y_pred, s=12, alpha=0.6)
    plt.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=1)
    plt.xlim(lim_min, lim_max); plt.ylim(lim_min, lim_max)
    plt.title(title)
    plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

def timeseries_overlay(y_true, y_pred, names, out_path, max_points=500):
    """Overlay GT vs Pred for each dimension (first max_points samples)."""
    T = min(len(y_true), max_points)
    t = np.arange(T)

    fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    for i, name in enumerate(names):
        axs[i].plot(t, y_true[:T, i], label='GT')
        axs[i].plot(t, y_pred[:T, i], label='Pred', alpha=0.8)
        axs[i].set_ylabel(name)
        axs[i].grid(True, alpha=0.25)
    axs[-1].set_xlabel('Sample (test split order)')
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

# 1) Save CSV of test predictions (robot units)
pred_df = pd.DataFrame(
    np.column_stack([y_true_robot, y_pred_robot]),
    columns=[
        "u_true_m", "v_true_m", "force_true_N",
        "u_pred_m", "v_pred_m", "force_pred_N"
    ]
)
csv_path = Path(OUT_DIR, "test_predictions_robot.csv")
pred_df.to_csv(csv_path, index=False)
print(f"Saved predictions CSV -> {csv_path}")

# # 2) Parity plots (robot units)
# parity_plot(
#     y_true=y_true_robot[:, 2], y_pred=y_pred_robot[:, 2],
#     title="Force (N): Prediction vs Ground Truth",
#     xlabel="Ground Truth (N)", ylabel="Prediction (N)",
#     out_path=Path(OUT_DIR, "parity_force_robot.pdf")
# )
# parity_plot(
#     y_true=y_true_robot[:, 0], y_pred=y_pred_robot[:, 0],
#     title="X position u (m): Prediction vs Ground Truth",
#     xlabel="Ground Truth u (m)", ylabel="Prediction u (m)",
#     out_path=Path(OUT_DIR, "parity_u_robot.pdf")
# )
# parity_plot(
#     y_true=y_true_robot[:, 1], y_pred=y_pred_robot[:, 1],
#     title="Y position v (m): Prediction vs Ground Truth",
#     xlabel="Ground Truth v (m)", ylabel="Prediction v (m)",
#     out_path=Path(OUT_DIR, "parity_v_robot.pdf")
# )
# print(f"Saved parity plots -> {OUT_DIR}")

# # 3) Optional: time-series overlay on test split (robot units)
# timeseries_overlay(
#     y_true=y_true_robot,
#     y_pred=y_pred_robot,
#     names=["u (m)", "v (m)", "force (N)"],
#     out_path=Path(OUT_DIR, "timeseries_overlay_robot.pdf"),
#     max_points=600  # adjust if you want more/less
# )
# print(f"Saved time-series overlay -> {OUT_DIR}")

label_names = ["x", "y", "force"]  # order: 0->x (u), 1->y (v), 2->force

for i, lbl in enumerate(label_names):
    plt.figure(figsize=(4.5, 4))
    plt.scatter(y_true_robot[:, i], y_pred_robot[:, i], alpha=0.7)
    plt.xlabel(f"True {lbl}", fontsize=16)
    plt.ylabel(f"Predicted {lbl}", fontsize=16)
    # plt.title(f"Physics-informed MLP: {lbl}")
    # plt.grid(True)
    plt.axis("equal")
    lo = float(np.min(y_true_robot[:, i]))
    hi = float(np.max(y_true_robot[:, i]))
    plt.plot([lo, hi], [lo, hi], 'r--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"phymlp_pred_{lbl}.pdf"), dpi=300, bbox_inches="tight")
    plt.close()

print(f"Saved physics-informed parity plots -> {OUT_DIR}")


# ===========================
# Step 8: Spatial force error map (for intuition)
# ===========================
# x_coords = y_true_robot[:, 0]
# y_coords = y_true_robot[:, 1]
# force_true = y_true_robot[:, 2]
# force_pred = y_pred_robot[:, 2]
# force_error = np.abs(force_pred - force_true)

# plt.figure(figsize=(4.6, 4.8))
# sc = plt.scatter(x_coords, y_coords, c=force_error, cmap='hot', s=60, edgecolors='k')
# plt.colorbar(sc, label="Force prediction error (N)", fraction=0.045, pad=0.05)
# plt.xlabel("x [m]", fontsize=16)
# plt.ylabel("y [m]", fontsize=16)
# # plt.title("MLP Force Prediction Error by Touch Location")
# plt.gca().set_aspect("equal")
# # plt.grid(True)
# plt.savefig(os.path.join(OUT_DIR, "force_error_map.pdf"), dpi=300)
# plt.close()
# print(f"📍 Saved spatial force error heatmap → {os.path.join(OUT_DIR, 'force_error_map.pdf')}")

x_coords = y_true_robot[:, 0]
y_coords = y_true_robot[:, 1]
force_true = y_true_robot[:, 2]
force_pred = y_pred_robot[:, 2]
force_error = np.abs(force_pred - force_true)

# Compute y-axis ticks: min, mid, max
ymin, ymax = float(np.min(y_coords)), float(np.max(y_coords))
ymid = (ymin + ymax) / 2.0
# yticks = [ymin, ymid, ymax]
yticks = [-0.03, -0.01, 0.01, 0.03]

fig, ax = plt.subplots(figsize=(4.6, 4.8))  # smaller figure
sc = ax.scatter(
    x_coords, y_coords,
    c=force_error, cmap='hot',
    s=40, edgecolors='k', linewidths=0.3
)

# Axes labels
ax.set_xlabel("x [m]", fontsize=16)
ax.set_ylabel("y [m]", fontsize=16)
ax.set_aspect("equal")

# Y ticks: show only min/mid/max and rotate 90°
ax.set_yticks(yticks)
ax.set_yticklabels([f"{t:.2f}" for t in yticks])
# for lab in ax.get_yticklabels():
#     lab.set_rotation(90)
#     lab.set_va('center')
#     lab.set_ha('center')

# Make colorbar same height, closer; no label
# - 'fraction' controls bar height relative to the axes (smaller -> thinner bar)
# - 'pad' controls spacing between bar and axes
cbar = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.05)
cbar.ax.set_ylabel("Force prediction error (N)", fontsize=14)
cbar.ax.tick_params(labelsize=12)
# Keep only first and last ticks on colorbar and rotate 90°
# vmin, vmax = ax.collections[0].get_clim()
# cbar.set_ticks([vmin, vmax])
# cbar.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])
# for t in cbar.ax.get_yticklabels():
#     t.set_rotation(90)
#     t.set_va('center')

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "force_error_map_small.pdf"), dpi=300, bbox_inches="tight")
plt.close()
print(f"📍 Saved compact spatial error map → {os.path.join(OUT_DIR, 'force_error_map_small.pdf')}")
