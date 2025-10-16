# reconstruct_samples_bp_sim.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- PyEIT ---
from pyeit.mesh import create as create_mesh
from pyeit.eit.protocol import create as create_protocol
from pyeit.eit.fem import EITForward
from pyeit.eit.bp import BP

# ===========================
# Config
# ===========================
DATA_PATH   = "/home/kiyanoush/Projects/eit_hybrid_state_estimation/data/touch_force_dataset_sim2real.csv"
OUT_DIR     = "recon_samples_bp_sim"
N_EL        = 16
TOPK        = 100                 # how many high-force samples to render
CMAP        = "viridis"
CLIP_PCT    = 99.0               # clip color scale (two-sided later)

os.makedirs(OUT_DIR, exist_ok=True)

# ===========================
# 1) Load sim CSV (Δv | x | y | force)
# ===========================
df = pd.read_csv(DATA_PATH)

# Voltages are all columns except last 3 (x,y,force)
volt_cols = df.columns[:-3]
assert len(volt_cols) > 0, "No voltage columns found in sim CSV."

X_dv = df[volt_cols].astype(np.float32).to_numpy()   # Δv
xyF  = df.iloc[:, -3:].to_numpy(np.float32)          # [x, y, force]
force = xyF[:, 2]

M = X_dv.shape[1]
print(f"[Sim] samples={X_dv.shape[0]}  M={M}")

# ===========================
# 2) Build nominal baseline v0 via PyEIT
# ===========================
mesh = create_mesh(n_el=N_EL, h0=0.1)
protocol = create_protocol(n_el=N_EL, dist_exc=1, step_meas=1, parser_meas="std")
fwd = EITForward(mesh, protocol)
v0 = fwd.solve_eit(mesh.perm).astype(np.float32)   # shape [M]
if v0.size != M:
    raise RuntimeError(f"Protocol length mismatch: v0 has {v0.size}, CSV has {M}")

# ===========================
# 3) BP setup + helpers
# ===========================
bp = BP(mesh, protocol)
bp.setup(weight="none")

def bp_node_to_element(ds_node: np.ndarray) -> np.ndarray:
    return ds_node[mesh.element].mean(axis=1).astype(np.float32)

def plot_recon(ds_elem, title, out_png, vmin=None, vmax=None):
    plt.figure(figsize=(4.2, 4.2))
    tpc = plt.tripcolor(
        mesh.node[:, 0], mesh.node[:, 1], mesh.element,
        ds_elem, shading="flat", cmap=CMAP, vmin=vmin, vmax=vmax
    )
    circ = plt.Circle((0, 0), radius=1.0, fill=False, color="k", linewidth=0.8, alpha=0.6)
    ax = plt.gca()
    ax.add_patch(circ)
    ax.set_aspect("equal")
    ax.axis("off")
    cb = plt.colorbar(tpc, fraction=0.046, pad=0.0, location="left")
    cb.ax.tick_params(labelsize=10, rotation=-45)
    plt.title(title, fontsize=16)
    plt.tight_layout(pad=0.1)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

def plot_voltage_signal(v_abs, title, out_png):
    fig, ax = plt.subplots(1, 1, figsize=(4, 2.5), sharex=True)
    ax.plot(v_abs, alpha=0.99, color='b')
    ax.set_ylabel("Voltage (a.u.)", fontsize=15)
    ax.set_xlabel("Measurement index", fontsize=15)
    # rotate y tick labels vertical to save width
    plt.setp(ax.get_yticklabels(), rotation=90, ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

# ===========================
# 4) Pick top-K high-force rows & reconstruct
# ===========================
idx_sorted = np.argsort(-force)   # descending by force
idx_sel = idx_sorted[:min(TOPK, len(idx_sorted))]
elem_maps = []
titles    = []

for j, i in enumerate(idx_sel):
    dv = X_dv[i]
    v1 = v0 + dv

    # BP solve
    ds_node = bp.solve(v1, v0, normalize=True, log_scale=False).astype(np.float32)
    if ds_node.shape[0] == mesh.node.shape[0] and ds_node.shape[0] != mesh.element.shape[0]:
        ds_elem = bp_node_to_element(ds_node)
    elif ds_node.shape[0] == mesh.element.shape[0]:
        ds_elem = ds_node
    else:
        raise RuntimeError(f"Unexpected BP length: {ds_node.shape[0]}")

    elem_maps.append(ds_elem)
    f_val = force[i]

    # --- save recon with force title
    recon_png = Path(OUT_DIR, f"sim_bp_touch_{j:02d}_recon.png")
    plot_recon(ds_elem, title=f"Force = {f_val:.2f} N", out_png=recon_png, vmin=vmin, vmax=vmax)

    # --- save voltage with force title
    volt_png = Path(OUT_DIR, f"sim_bp_touch_{j:02d}_volt.png")
    plot_voltage_signal(v1, title=f"Force = {f_val:.2f} N", out_png=volt_png)

    print(f"Saved touch #{j:02d} → {recon_png.name}, {volt_png.name}")

elem_maps = np.stack(elem_maps, axis=0)

# consistent, *symmetric* color scale across selected panels
hi = np.percentile(np.abs(elem_maps), CLIP_PCT)
vmin, vmax = -hi, hi

# save recon + voltage for each selected sample
for j, i in enumerate(idx_sel):
    recon_png = Path(OUT_DIR, f"sim_bp_touch_{j:02d}_recon.png")
    plot_recon(elem_maps[j], titles[j], recon_png, vmin=vmin, vmax=vmax)

    dv = X_dv[i]
    v1 = v0 + dv
    volt_png = Path(OUT_DIR, f"sim_bp_touch_{j:02d}_volt.png")
    plot_voltage_signal(v1, titles[j], volt_png)

    print(f"Saved touch #{j:02d} → {recon_png.name}, {volt_png.name}")

# ===========================
# 5) Also render a no-touch exemplar
# ===========================
plot_recon(
    ds_elem=np.zeros(mesh.element.shape[0], dtype=np.float32),
    title="Force = 0.00 N (no-touch)",
    out_png=Path(OUT_DIR, "sim_bp_notouch_recon.png"),
    vmin=vmin, vmax=vmax
)
plot_voltage_signal(
    v_abs=v0,
    title="Force = 0.00 N (no-touch)",
    out_png=Path(OUT_DIR, "sim_bp_notouch_volt.png")
)

print(f"✅ Saved no-touch exemplars in {OUT_DIR}/")
