import numpy as np
import csv
import os
import copy
import pyeit.mesh as mesh
from pyeit.mesh import set_perm
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
from pyeit.eit.protocol import create as create_protocol
from pyeit.eit.fem import EITForward
import matplotlib.pyplot as plt

# ===========================
# Controls / knobs
# ===========================
OUT_FILE = "/home/kiyanoush/Projects/eit_hybrid_state_estimation/data/touch_force_dataset_sim2real.csv"
PLOT_FILE = "/home/kiyanoush/Projects/eit_hybrid_state_estimation/data/touch_locations_plot_sim2real.png"
os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

RNG_SEED = 7
np.random.seed(RNG_SEED)

n_el = 16
touch_radius = 0.10

# ---- Force sampling ----
n_force_samples = 100
force_levels = np.clip(np.random.normal(loc=0.5, scale=0.2, size=n_force_samples), 0.0, 1.0)
force_levels = np.unique(np.round(force_levels, 3))
perm_levels  = 1.0 + 9.0 * force_levels  # same mapping as your script (can make nonlinear later)

# ---- Touch grid (polar) ----
n_radii = 8
n_angles = 24
r_vals = np.linspace(0.1, 1.0 - touch_radius, n_radii)
theta_vals = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
all_touch_points = [(r*np.cos(t), r*np.sin(t)) for r in r_vals for t in theta_vals]

# ===========================
# Distortion model toggles
# ===========================
USE_GEOMETRY_MISMATCH   = True   # elliptical plant mesh + electrode angle jitter
USE_CONTACT_IMP_VARIAB  = True   # per-channel gains (mirror-paired)
USE_CROSSTALK           = True   # near-diagonal mixing matrix
USE_MEAS_NOISE          = True   # additive Gaussian noise
USE_FORCE_DEP_CONTACT   = True   # optional: gains drift slightly with force

# Strengths
ELLIPSE_SCALE_STD   = 0.03   # std of ellipse scale deltas (sx, sy) ~ N(0, std)
ELECTRODE_ANG_STD   = 0.5*np.pi/180.0  # radians, small jitter per electrode
CONTACT_GAIN_STD    = 0.04   # std of log-normal-ish gain (we’ll use N(0, std) on multiplicative factor)
CROSSTALK_EPS       = 0.03   # magnitude of neighbor coupling
NOISE_REL_STD       = 0.01   # noise std relative to median |v| of (baseline or sample)
FORCE_GAIN_COEF     = 0.05   # if enabled, gain drifts by ~coef*force around the base gain

# Session controls
SESSION_SIZE         = 300    # how many (x,y,force) samples per “session” (shared distortions)
N_SESSIONS           = None   # None = inferred from total samples
PRINT_PROGRESS_EVERY = 200

# ===========================
# Helpers
# ===========================
def build_base_mesh_and_protocol(n_el=16):
    m = mesh.create(n_el=n_el)
    p = create_protocol(n_el=n_el, dist_exc=1, step_meas=1)  # std parser by default
    return m, p

def make_elliptical_mesh(base_mesh, sx=1.0, sy=1.0, electrode_angle_jitter=None):
    """Return a deep-copied mesh transformed to an ellipse and with (optional) electrode angle jitter."""
    m = copy.deepcopy(base_mesh)
    # Elliptical boundary via scaling of node xy
    m.node[:, 0] *= sx
    m.node[:, 1] *= sy
    # Electrode jitter (angles) – approximate by rotating electrode nodes slightly around origin
    if electrode_angle_jitter is not None and len(m.el_pos) == len(electrode_angle_jitter):
        el_idx = m.el_pos.copy()
        # For each electrode index, perturb its node angle a bit
        for k, nd in enumerate(el_idx):
            xy = m.node[nd, :2].copy()
            ang = np.arctan2(xy[1], xy[0]) + electrode_angle_jitter[k]
            r   = np.linalg.norm(xy)
            xy2 = np.array([r*np.cos(ang), r*np.sin(ang)], dtype=np.float64)
            m.node[nd, 0] = xy2[0]
            m.node[nd, 1] = xy2[1]
    return m

def make_crosstalk_matrix(M, eps=CROSSTALK_EPS):
    """
    Simple wrap-around tridiagonal mixing:
    C = (1-2eps)I + eps*(shift_left + shift_right)
    """
    if eps <= 0:
        return np.eye(M, dtype=np.float32)
    C = np.eye(M, dtype=np.float64)*(1.0 - 2.0*eps)
    for i in range(M):
        C[i, (i-1) % M] += eps
        C[i, (i+1) % M] += eps
    return C.astype(np.float32)

def make_mirror_paired_gains(M, std=CONTACT_GAIN_STD, rng=np.random):
    """
    Channel-wise multiplicative gains g (~1 +/- std), mirror-paired:
    g[i] = g[M-1-i]
    """
    g = np.ones(M, dtype=np.float64)
    half = (M+1)//2
    for i in range(half):
        delta = rng.normal(loc=0.0, scale=std)
        g[i] = 1.0 + delta
        g[M-1-i] = g[i]
    return g.astype(np.float32)

def apply_session_distortions(v, g, C):
    """
    Apply contact impedance gains and crosstalk:
    v' = C @ (g ⊙ v)
    """
    return (C @ (g * v)).astype(np.float32)

def add_measurement_noise(v, rel_std=NOISE_REL_STD, rng=np.random):
    """
    Additive Gaussian noise with std proportional to median |v|.
    """
    scale = np.median(np.abs(v)) + 1e-12
    sigma = rel_std * scale
    noise = rng.normal(0.0, sigma, size=v.shape).astype(np.float32)
    return (v + noise).astype(np.float32)

# ===========================
# Build “reconstruction” mesh & protocol (nominal)
# ===========================
mesh_nom, protocol_nom = build_base_mesh_and_protocol(n_el=n_el)
fwd_nom = EITForward(mesh_nom, protocol_nom)

# For reference (baseline) on nominal mesh – not used directly when geometry mismatch is ON,
# but we’ll keep it for clarity and potential use.
v_baseline_nom = fwd_nom.solve_eit(mesh_nom.perm).astype(np.float32)
M = v_baseline_nom.size

# ===========================
# Prepare sessions (simulate hardware biases consistent over time)
# ===========================
total_samples = len(all_touch_points) * len(force_levels)
if N_SESSIONS is None:
    N_SESSIONS = max(1, int(np.ceil(total_samples / SESSION_SIZE)))

# Pre-sample session distortions
sessions = []
for s in range(N_SESSIONS):
    # Geometry mismatch params
    if USE_GEOMETRY_MISMATCH:
        sx = 1.0 + np.random.normal(0.0, ELLIPSE_SCALE_STD)
        sy = 1.0 + np.random.normal(0.0, ELLIPSE_SCALE_STD)
        el_jit = np.random.normal(0.0, ELECTRODE_ANG_STD, size=n_el)
        mesh_plant = make_elliptical_mesh(mesh_nom, sx=sx, sy=sy, electrode_angle_jitter=el_jit)
    else:
        mesh_plant = copy.deepcopy(mesh_nom)

    fwd_plant = EITForward(mesh_plant, protocol_nom)
    v0_plant  = fwd_plant.solve_eit(mesh_plant.perm).astype(np.float32)

    # Contact impedance variability (channel gains, mirror-paired)
    g = make_mirror_paired_gains(M, std=CONTACT_GAIN_STD, rng=np.random) if USE_CONTACT_IMP_VARIAB else np.ones(M, np.float32)

    # Crosstalk matrix
    C = make_crosstalk_matrix(M, eps=CROSSTALK_EPS) if USE_CROSSTALK else np.eye(M, dtype=np.float32)

    sessions.append(dict(mesh=mesh_plant, fwd=fwd_plant, v0=v0_plant, g=g, C=C))

# ===========================
# Generate & write CSV
# ===========================
with open(OUT_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    header = [f"v{i}" for i in range(M)] + ["x", "y", "force"]
    writer.writerow(header)

    sample_idx = 0
    for (x, y) in all_touch_points:
        for force, perm in zip(force_levels, perm_levels):
            sess_id = min(sample_idx // SESSION_SIZE, N_SESSIONS - 1)
            sess = sessions[sess_id]
            fwd_plant = sess["fwd"]
            v0 = sess["v0"]
            g  = sess["g"].copy()
            C  = sess["C"]

            # Optional: make contact gains drift slightly with force
            if USE_FORCE_DEP_CONTACT and USE_CONTACT_IMP_VARIAB:
                g = (g * (1.0 + FORCE_GAIN_COEF*(force - 0.5))).astype(np.float32)

            # “True” plant anomaly and voltages
            anomaly = [PyEITAnomaly_Circle(center=[x, y], r=touch_radius, perm=perm)]
            mesh_mod = set_perm(sess["mesh"], anomaly=anomaly)
            v_touch  = fwd_plant.solve_eit(mesh_mod.perm).astype(np.float32)

            # Apply consistent session distortions to both baseline and touch
            v0_d     = apply_session_distortions(v0, g, C)
            v_touch_d= apply_session_distortions(v_touch, g, C)

            # Add measurement noise independently to each reading (like separate acquisitions)
            if USE_MEAS_NOISE:
                v0_d     = add_measurement_noise(v0_d, rel_std=NOISE_REL_STD, rng=np.random)
                v_touch_d= add_measurement_noise(v_touch_d, rel_std=NOISE_REL_STD, rng=np.random)

            delta_v = (v_touch_d - v0_d).astype(np.float32)
            row = list(delta_v) + [x, y, float(force)]
            writer.writerow(row)

            sample_idx += 1
            if (sample_idx % PRINT_PROGRESS_EVERY) == 0:
                print(f"Saved {sample_idx}/{total_samples} samples...")

print(f"\n✅ Dataset saved to: {OUT_FILE}")

# ===========================
# Plot mesh and touch points (nominal geometry shown)
# ===========================
pts = mesh_nom.node
el_pos = mesh_nom.el_pos
touch_points = np.array(all_touch_points)

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title("Synthetic Touch Locations on EIT Sensor (sim2real)")
ax.set_aspect("equal")
ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
ax.plot(touch_points[:, 0], touch_points[:, 1], 'bo', ms=3, label="Touch locations")
ax.plot(pts[el_pos, 0], pts[el_pos, 1], 'ro', label="Electrodes")
circle = plt.Circle((0, 0), 1.0, color='gray', fill=False, linestyle='--')
ax.add_patch(circle)
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(PLOT_FILE, dpi=300)
plt.close()
print(f"✅ Touch location map saved as '{PLOT_FILE}'")
