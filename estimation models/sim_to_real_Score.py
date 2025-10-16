# # simreal_similarity_shape_aware_fair_fixed.py
# import os
# import numpy as np
# from scipy.signal import correlate
# from scipy.fft import rfft
# from scipy.spatial.distance import cdist
# from scipy.stats import wasserstein_distance, spearmanr

# IN_DIR = "eda_voltage_plots"

# # ---------------- Load arrays ----------------
# real  = np.load(os.path.join(IN_DIR, "real_no_touch_voltages.npy"))       # [Nr, M]
# plain = np.load(os.path.join(IN_DIR, "plain_sim_no_touch_voltage.npy"))   # [M]
# pert  = np.load(os.path.join(IN_DIR, "pert_sim_no_touch_voltages.npy"))   # [Np, M]

# plain = plain[None, :]  # -> [1, M]

# # ---------------- Helpers ----------------
# def zscore_to_real(X, mu, sigma, eps=1e-8):
#     return (X - mu[None, :]) / (sigma[None, :] + eps)

# def mmd_rbf(X, Y, gamma):
#     def kxx(A):
#         AA = np.sum(A*A, 1, keepdims=True)
#         D  = AA + AA.T - 2*A@A.T
#         K  = np.exp(-gamma*D)
#         n  = len(A)
#         return (K.sum() - np.trace(K)) / max(n*(n-1), 1)
#     Kxx = kxx(X)
#     Kyy = kxx(Y)
#     Kxy = np.exp(-gamma * cdist(X, Y, 'sqeuclidean')).mean()
#     return Kxx + Kyy - 2*Kxy

# def coral_gap_shrink(X, Y, alpha=1e-3):
#     Xc = X - X.mean(0, keepdims=True)
#     Yc = Y - Y.mean(0, keepdims=True)
#     nX = max(len(X)-1, 1)
#     nY = max(len(Y)-1, 1)
#     Cx = (Xc.T @ Xc) / nX
#     Cy = (Yc.T @ Yc) / nY
#     d  = X.shape[1]
#     Cx = (1-alpha)*Cx + alpha*np.eye(d)
#     Cy = (1-alpha)*Cy + alpha*np.eye(d)
#     return np.linalg.norm(Cx - Cy, ord='fro')

# def best_circular_ncc(x, y):
#     x = (x - x.mean())/(x.std()+1e-12)
#     y = (y - y.mean())/(y.std()+1e-12)
#     c = correlate(x, y, mode='full', method='fft')
#     return float(c.max()/(len(x)+1e-12))

# def spectral_cosine(x, y):
#     x = x - x.mean(); y = y - y.mean()
#     X = np.abs(rfft(x))**2; Y = np.abs(rfft(y))**2
#     X = X/(X.sum()+1e-12); Y = Y/(Y.sum()+1e-12)
#     num = float((X*Y).sum())
#     den = float(np.sqrt((X*X).sum())*np.sqrt((Y*Y).sum()) + 1e-12)
#     return num/den

# def energy_distance(X, Y):
#     dxy = cdist(X, Y).mean()
#     dxx = cdist(X, X).mean()
#     dyy = cdist(Y, Y).mean()
#     return 2*dxy - dxx - dyy

# def sliced_wasserstein(X, Y, n_projs=64, q_points=256, seed=0):
#     """
#     SWD between multivariate sets X,Y using 1D projections and
#     quantile functions on a shared grid (handles different sample sizes).
#     """
#     rng = np.random.default_rng(seed)
#     d = X.shape[1]
#     qs = np.linspace(0.0, 1.0, q_points)
#     sw = 0.0
#     for _ in range(n_projs):
#         w = rng.normal(size=d); w = w/np.linalg.norm(w)
#         x1 = X @ w
#         y1 = Y @ w
#         # compare quantiles (L1 distance between quantile functions)
#         qx = np.quantile(x1, qs)
#         qy = np.quantile(y1, qs)
#         sw += float(np.mean(np.abs(qx - qy)))
#     return sw / n_projs

# def replicate_with_jitter(X1, n=32, rel_jit=0.01, seed=0):
#     rng = np.random.default_rng(seed)
#     base = np.repeat(X1, n, axis=0)
#     g = rng.normal(1.0, rel_jit, size=base.shape)
#     return base * g

# # ---------------- Fairify PlainSim ----------------
# target_n = max(len(pert), 16)
# plain_rep = replicate_with_jitter(plain, n=target_n, rel_jit=0.01, seed=123)

# # ---------------- Z-score to Real ----------------
# mu_r = real.mean(0); sd_r = real.std(0) + 1e-6
# real_z  = zscore_to_real(real,       mu_r, sd_r)
# plain_z = zscore_to_real(plain_rep,  mu_r, sd_r)
# pert_z  = zscore_to_real(pert,       mu_r, sd_r)

# # Gamma for MMD via median heuristic on Real
# d2 = cdist(real_z, real_z, 'sqeuclidean')
# med = np.median(d2[d2>0]) if np.any(d2>0) else 1.0
# gamma = 1.0/max(med, 1e-6)

# def report(tag, A, B):
#     mmd   = mmd_rbf(A, B, gamma)
#     coral = coral_gap_shrink(A, B, alpha=1e-3)
#     ncc   = best_circular_ncc(A.mean(0), B.mean(0))
#     spec  = spectral_cosine(A.mean(0), B.mean(0))
#     emd   = wasserstein_distance(A.ravel(), B.ravel())   # pooled 1D distance after z-score
#     enrg  = energy_distance(A, B)
#     swd   = sliced_wasserstein(A, B, n_projs=64, q_points=256, seed=7)
#     r, _  = spearmanr(A.mean(0), B.mean(0))
#     spearman_ch = float(0.0 if np.isnan(r) else r)

#     print(f"[{tag}]  "
#           f"MMD(z)↓={mmd:.4f} | CORAL(z)↓={coral:.2f} | "
#           f"NCC↑={ncc:.3f} | SpecCos↑={spec:.3f} | "
#           f"EMD(z)↓={emd:.3f} | Energy↓={enrg:.3f} | SWD↓={swd:.3f} | "
#           f"Spearman(μ_ch)↑={spearman_ch:.3f}")

# print("\n=== Shape-aware, scale-invariant similarity (no-touch only; fair comparison) ===")
# report("PlainSim vs Real", plain_z, real_z)
# report("PertSim  vs Real", pert_z,   real_z)



import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---- PyEIT imports ----
import pyeit.mesh as mesh
from pyeit.eit.protocol import create as create_protocol
from pyeit.eit.fem import EITForward

# ===============================
# Paths (update as needed)
# ===============================
REAL_PATH   = "data/merged_with_probe.csv"  # real dataset with EIT columns + force_n
OUT_DIR     = "eda_voltage_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# If you want to *also* save arrays for later analysis:
SAVE_NUMPY  = True

# ===============================
# Controls / knobs (match your sim2real generator)
# ===============================
RNG_SEED = 7
np.random.seed(RNG_SEED)

n_el = 16  # electrodes
# Distortions
USE_GEOMETRY_MISMATCH   = True
USE_CONTACT_IMP_VARIAB  = True
USE_CROSSTALK           = True
USE_MEAS_NOISE          = True

# ELLIPSE_SCALE_STD   = 0.03
# ELECTRODE_ANG_STD   = 0.5*np.pi/180.0
# CONTACT_GAIN_STD    = 0.04
# CROSSTALK_EPS       = 0.03
# NOISE_REL_STD       = 0.01

ELLIPSE_SCALE_STD   = 0.05
ELECTRODE_ANG_STD   = 0.5*np.pi/180.0
CONTACT_GAIN_STD    = 0.08
CROSSTALK_EPS       = 0.06
NOISE_REL_STD       = 0.02

# How many perturbed sessions (independent baselines) to draw
N_PERT_SESSIONS = 8

# How many random curves to draw from each source in the figure
N_SAMPLES = 1

# Real-data baseline threshold
REAL_BASELINE_THR = 0.05  # force_n < 0.05

# ===============================
# Helpers
# ===============================
def build_base_mesh_and_protocol(n_el=16):
    m = mesh.create(n_el=n_el)
    p = create_protocol(n_el=n_el, dist_exc=1, step_meas=1)  # "std" parser by default
    return m, p

def make_elliptical_mesh(base_mesh, sx=1.0, sy=1.0, electrode_angle_jitter=None):
    """Deep-copy mesh, apply ellipse scaling and slight electrode angle jitter."""
    m = copy.deepcopy(base_mesh)
    m.node[:, 0] *= sx
    m.node[:, 1] *= sy
    if electrode_angle_jitter is not None and len(m.el_pos) == len(electrode_angle_jitter):
        el_idx = m.el_pos.copy()
        for k, nd in enumerate(el_idx):
            xy = m.node[nd, :2].copy()
            ang = np.arctan2(xy[1], xy[0]) + electrode_angle_jitter[k]
            r   = np.linalg.norm(xy)
            xy2 = np.array([r*np.cos(ang), r*np.sin(ang)], dtype=np.float64)
            m.node[nd, 0] = xy2[0]
            m.node[nd, 1] = xy2[1]
    return m

def make_mirror_paired_gains(M, std=CONTACT_GAIN_STD, rng=np.random):
    """Channel-wise multiplicative gains g (~1 +/- std), mirror-paired: g[i] == g[M-1-i]."""
    g = np.ones(M, dtype=np.float64)
    half = (M + 1) // 2
    for i in range(half):
        delta = rng.normal(loc=0.0, scale=std)
        g[i] = 1.0 + delta
        g[M-1-i] = g[i]
    return g.astype(np.float32)

def make_crosstalk_matrix(M, eps=CROSSTALK_EPS):
    """Wrap-around tridiagonal mixing matrix."""
    if eps <= 0:
        return np.eye(M, dtype=np.float32)
    C = np.eye(M, dtype=np.float64)*(1.0 - 2.0*eps)
    for i in range(M):
        C[i, (i-1) % M] += eps
        C[i, (i+1) % M] += eps
    return C.astype(np.float32)

def apply_session_distortions(v, g, C):
    """v' = C @ (g ⊙ v)."""
    return (C @ (g * v)).astype(np.float32)

def add_measurement_noise(v, rel_std=NOISE_REL_STD, rng=np.random):
    """Additive Gaussian noise with std proportional to median |v|."""
    scale = np.median(np.abs(v)) + 1e-12
    sigma = rel_std * scale
    noise = rng.normal(0.0, sigma, size=v.shape).astype(np.float32)
    return (v + noise).astype(np.float32)

# ===============================
# 1) Real no-touch baselines (raw)
# ===============================
real_df = pd.read_csv(REAL_PATH)

# Identify EIT columns, e.g. "eit_0", "eit_1", ...
eit_cols = [c for c in real_df.columns if c.lower().startswith("eit_")]
if len(eit_cols) == 0:
    raise RuntimeError("No EIT columns found in the real CSV (expected columns like 'eit_0', 'eit_1', ...).")

if "force_n" not in real_df.columns:
    raise RuntimeError("Real CSV must contain 'force_n' column to select no-touch baselines.")

# Mask for no-touch rows
mask_baseline = real_df["force_n"].notna() & (real_df["force_n"] < REAL_BASELINE_THR)

# Extract baseline voltages
real_baselines = real_df.loc[mask_baseline, eit_cols].to_numpy(dtype=np.float32)

# --- NEW STEP: remove channels that are always zero ---
nonzero_cols = np.any(real_baselines != 0.0, axis=0)   # True if channel has any nonzero value
real_baselines = real_baselines[:, nonzero_cols]

# Also update eit_cols list to keep only nonzero channels
eit_cols = [c for c, keep in zip(eit_cols, nonzero_cols) if keep]

print(f"[Real] baseline rows: {real_baselines.shape[0]} | kept channels: {real_baselines.shape[1]}")

# ===============================
# 2) Plain-sim baseline
# ===============================
mesh_nom, protocol_nom = build_base_mesh_and_protocol(n_el=n_el)
fwd_nom = EITForward(mesh_nom, protocol_nom)
v0_plain = fwd_nom.solve_eit(mesh_nom.perm).astype(np.float32)  # single baseline vector (no distortions)
M = v0_plain.size
print(f"[PlainSim] baseline shape: ({M},)")

# ===============================
# 3) Perturbed-sim baselines (multiple sessions)
# ===============================
pert_baselines = []
for s in range(N_PERT_SESSIONS):
    # Geometry mismatch
    if USE_GEOMETRY_MISMATCH:
        sx = 1.0 + np.random.normal(0.0, ELLIPSE_SCALE_STD)
        sy = 1.0 + np.random.normal(0.0, ELLIPSE_SCALE_STD)
        el_jit = np.random.normal(0.0, ELECTRODE_ANG_STD, size=n_el)
        mesh_plant = make_elliptical_mesh(mesh_nom, sx=sx, sy=sy, electrode_angle_jitter=el_jit)
    else:
        mesh_plant = copy.deepcopy(mesh_nom)
    fwd_plant = EITForward(mesh_plant, protocol_nom)
    v0 = fwd_plant.solve_eit(mesh_plant.perm).astype(np.float32)

    # Session distortions
    g = make_mirror_paired_gains(M, std=CONTACT_GAIN_STD, rng=np.random) if USE_CONTACT_IMP_VARIAB else np.ones(M, np.float32)
    C = make_crosstalk_matrix(M, eps=CROSSTALK_EPS) if USE_CROSSTALK else np.eye(M, dtype=np.float32)

    v0_d = apply_session_distortions(v0, g, C)
    if USE_MEAS_NOISE:
        v0_d = add_measurement_noise(v0_d, rel_std=NOISE_REL_STD, rng=np.random)

    pert_baselines.append(v0_d)

pert_baselines = np.stack(pert_baselines, axis=0)  # (N_PERT_SESSIONS, M)
print(f"[PertSim] baselines: {pert_baselines.shape}")

# ===============================
# Plot: three stacked subplots
# ===============================
# Choose random indices for plotting
rng = np.random.default_rng(RNG_SEED)
idx_real = rng.choice(real_baselines.shape[0], size=min(N_SAMPLES, real_baselines.shape[0]), replace=False)

# fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig, axes = plt.subplots(1, 1, figsize=(4, 2.5), sharex=True)

# (1) Real baselines
for i in idx_real:
    axes.plot(real_baselines[i], alpha=0.99, color='b')
# axes.set_title("Real No-Touch Voltage", fontsize=14)
axes.set_ylabel("Voltage (a.u.)", fontsize=15)
axes.set_xlabel("Measurement index", fontsize=15)
# axes[0].grid(True, alpha=0.3)
plt.setp(axes.get_yticklabels(), rotation=90, ha="center", va="center")

# (2) Plain-sim baseline (repeat the single baseline a few times for visual context)
# for _ in range(min(N_SAMPLES, 5)):
#     axes[1].plot(v0_plain, alpha=0.9)
# axes[1].set_title("PlainSim No-Touch Voltage", fontsize=14)
# axes[1].set_ylabel("Voltage (a.u.)", fontsize=12)
# # axes[1].grid(True, alpha=0.3)

# (3) Perturbed-sim baselines
# idx_pert = rng.choice(pert_baselines.shape[0], size=min(N_SAMPLES, pert_baselines.shape[0]), replace=False)
# for i in idx_pert:
#     axes[1].plot(pert_baselines[i]*8.0, alpha=0.75)
# axes[1].set_title("PertSim No-Touch Voltage", fontsize=14)
# axes[1].set_xlabel("Measurement index", fontsize=12)
# axes[1].set_ylabel("Voltage (a.u.)", fontsize=12)
# # axes[2].grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "no_touch_voltage_comparison.pdf")
plt.savefig(out_path, dpi=300)
plt.close()
print(f"✅ Saved: {out_path}")

# ===============================
# (Optional) Save arrays for further analysis
# ===============================
if SAVE_NUMPY:
    np.save(os.path.join(OUT_DIR, "real_no_touch_voltages.npy"), real_baselines)
    np.save(os.path.join(OUT_DIR, "plain_sim_no_touch_voltage.npy"), v0_plain)
    np.save(os.path.join(OUT_DIR, "pert_sim_no_touch_voltages.npy"), pert_baselines)
    print(f"💾 Saved arrays to {OUT_DIR}/")
