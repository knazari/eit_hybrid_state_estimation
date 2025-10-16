# reconstruct_samples_bp.py
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

plt.rcParams['pdf.fonttype'] = 42   # use TrueType fonts, not Type 3
plt.rcParams['ps.fonttype'] = 42


# ===========================
# Config
# ===========================
DATA_PATH   = "data/merged_with_probe.csv"      # <- real dataset
OUT_DIR     = "recon_samples_bp"
N_EL        = 16                                # electrodes
FORCE_BASELINE_THR = 0.05                       # rows with force < thr = no-touch baseline
TOPK        = 200                                 # how many high-force samples to render
CMAP        = "viridis"                           # "magma", "inferno", etc.
CLIP_PCT    = 99.0                              # clip color scale to this percentile for readability

os.makedirs(OUT_DIR, exist_ok=True)

# ===========================
# 1) Load & pick columns
# ===========================
df = pd.read_csv(DATA_PATH)

# EIT channels: keep non-constant, exclude 't_eit'
eit_cols = [c for c in df.columns if c.lower().startswith("eit_") and c.lower() != "t_eit"]
if not eit_cols:
    raise RuntimeError("No EIT columns like 'eit_0'.. found in CSV.")

# Drop channels that are all-zero across dataset (robustness)
keep = []
for c in eit_cols:
    col = df[c].astype(np.float32).values
    if np.nanstd(col) > 1e-12 and np.any(col != 0.0):
        keep.append(c)
eit_cols = keep
if not eit_cols:
    raise RuntimeError("All candidate EIT channels were constant/zero.")

# Targets (optional for titles)
has_u = "contact_u_m" in df.columns
has_v = "contact_v_m" in df.columns
has_f = "force_n"     in df.columns
if not has_f:
    raise RuntimeError("Real CSV must contain 'force_n'.")

# ===========================
# 2) Baseline from no-touch
# ===========================
mask_baseline = df["force_n"].notna() & (df["force_n"] < FORCE_BASELINE_THR)
if mask_baseline.sum() == 0:
    raise RuntimeError("No 'no-touch' rows (force_n < threshold) to compute baseline.")

v_ref = df.loc[mask_baseline, eit_cols].astype(np.float32).to_numpy()
# Channel-wise mean baseline
v0 = np.nanmean(v_ref, axis=0).astype(np.float32)  # shape [M]

# ===========================
# 3) Select TOP-K high-force rows
# ===========================
mask_touch = df["force_n"].notna() & (df["force_n"] >= FORCE_BASELINE_THR)
df_touch = df.loc[mask_touch, :].copy()
df_touch = df_touch.sort_values("force_n", ascending=False).head(TOPK).reset_index(drop=True)

V_touch = df_touch[eit_cols].astype(np.float32).to_numpy()  # [K, M]

# ===========================
# 4) PyEIT setup (BP)
# ===========================
mesh = create_mesh(n_el=N_EL, h0=0.1)
protocol = create_protocol(n_el=N_EL, dist_exc=1, step_meas=1, parser_meas="std")
_ = EITForward(mesh, protocol)   # forward obj not directly used

bp = BP(mesh, protocol)
bp.setup(weight="none")          # matches your realtime choice

def bp_node_to_element(ds_node: np.ndarray) -> np.ndarray:
    """Average node values over triangle elements -> element-wise map."""
    return ds_node[mesh.element].mean(axis=1).astype(np.float32)

# ===========================
# 5) Reconstruct & plot
# ===========================
def plot_single(ds_elem, title, out_png, vmin=None, vmax=None):
    plt.figure(figsize=(4.2, 4.2))
    tpc = plt.tripcolor(
        mesh.node[:, 0], mesh.node[:, 1], mesh.element,
        ds_elem, shading="flat", cmap=CMAP, vmin=vmin, vmax=vmax
    )
    circ = plt.Circle((0, 0), radius=1.0, fill=False, color="k", linewidth=0.8, alpha=0.6)
    plt.gca().add_patch(circ)
    plt.gca().set_aspect("equal")
    plt.axis("off")
    cb = plt.colorbar(tpc, fraction=0.046, pad=0.0, location='left')
    cb.ax.tick_params(labelsize=10, rotation=-45)
    plt.title(title, fontsize=16)
    plt.tight_layout(pad=0.1)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

# Reconstruct all, collect for consistent color scale
elem_maps = []
titles    = []
for i in range(V_touch.shape[0]):
    v1 = V_touch[i].astype(np.float32)

    # BP reconstruction on nodes (normalize=True, log_scale=False)
    ds_node = bp.solve(v1, v0, normalize=True, log_scale=False).astype(np.float32)

    # If BP returns node-wise map, convert to element-wise
    if ds_node.shape[0] == mesh.node.shape[0] and ds_node.shape[0] != mesh.element.shape[0]:
        ds_elem = bp_node_to_element(ds_node)
    elif ds_node.shape[0] == mesh.element.shape[0]:
        ds_elem = ds_node
    else:
        raise RuntimeError(f"Unexpected BP length: {ds_node.shape[0]}")

    elem_maps.append(ds_elem)
    fN  = df_touch.loc[i, "force_n"] if has_f else None
    um  = df_touch.loc[i, "contact_u_m"] if has_u else None
    vm  = df_touch.loc[i, "contact_v_m"] if has_v else None
    if has_u and has_v and has_f:
        titles.append(f"F={fN:.2f} N")
        # titles.append(f"F={fN:.2f} N | (u,v)=({um:.3f},{vm:.3f}) m")
    # elif has_f:
    #     titles.append(f"F={fN:.2f} N")
    # else:
    #     titles.append(f"Sample {i}")

elem_maps = np.stack(elem_maps, axis=0)  # [K, n_elem]

# Consistent color scale across panels (clip to percentile for readability)
lo = np.percentile(elem_maps, 100 - CLIP_PCT)
hi = np.percentile(elem_maps, CLIP_PCT)

# Save individual panels
panel_paths = []
for i in range(elem_maps.shape[0]):
    png_path = Path(OUT_DIR, f"bp_sample_{i:02d}.pdf")
    plot_single(elem_maps[i], titles[i], png_path, vmin=lo, vmax=hi)
    panel_paths.append(png_path)
    print(f"Saved {png_path}")

# ===========================
# 6) Optional: grid montage
# ===========================
# cols = min(3, TOPK)
# rows = int(np.ceil(TOPK / cols))
# plt.figure(figsize=(4.2*cols, 4.2*rows))
# for i in range(TOPK):
#     plt.subplot(rows, cols, i+1)
#     tpc = plt.tripcolor(
#         mesh.node[:, 0], mesh.node[:, 1], mesh.element,
#         elem_maps[i], shading="flat", cmap=CMAP, vmin=lo, vmax=hi
#     )
#     circ = plt.Circle((0, 0), radius=1.0, fill=False, color="k", linewidth=0.8, alpha=0.6)
#     plt.gca().add_patch(circ)
#     plt.gca().set_aspect("equal"); plt.axis("off")
#     plt.title(titles[i], fontsize=9)
# plt.tight_layout(pad=0.1, w_pad=0.4, h_pad=0.4)
# grid_path = Path(OUT_DIR, f"bp_samples_grid_{TOPK}.pdf")
# plt.savefig(grid_path, dpi=300, bbox_inches="tight")
# plt.close()
# print(f"✅ Saved grid montage -> {grid_path}")


# ===========================
# 5b) Voltage plotting (like your baseline code)
# ===========================
def plot_voltage_signal(voltage, title, out_png):
    plt.figure(figsize=(4, 2.5))
    plt.plot(voltage, alpha=0.99, color="b")
    plt.ylabel("Voltage (a.u.)", fontsize=15)
    plt.xlabel("Measurement index", fontsize=15)
    plt.setp(plt.gca().get_yticklabels(), rotation=90, ha="center", va="center")
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()

# ===========================
# 6) Reconstruct + save both conductivity & voltages
# ===========================
elem_maps = []
titles    = []
for i in range(V_touch.shape[0]):
    v1 = V_touch[i].astype(np.float32)

    # BP reconstruction
    ds_node = bp.solve(v1, v0, normalize=True, log_scale=False).astype(np.float32)
    if ds_node.shape[0] == mesh.node.shape[0] and ds_node.shape[0] != mesh.element.shape[0]:
        ds_elem = bp_node_to_element(ds_node)
    elif ds_node.shape[0] == mesh.element.shape[0]:
        ds_elem = ds_node
    else:
        raise RuntimeError(f"Unexpected BP length: {ds_node.shape[0]}")

    elem_maps.append(ds_elem)
    fN  = df_touch.loc[i, "force_n"] if has_f else None
    if has_f:
        titles.append(f"F={fN:.2f} N")
    else:
        titles.append(f"Sample {i}")

# Convert for consistent color scale
elem_maps = np.stack(elem_maps, axis=0)
lo = np.percentile(elem_maps, 100 - CLIP_PCT)
hi = np.percentile(elem_maps, CLIP_PCT)

# Save panels
for i in range(elem_maps.shape[0]):
    # conductivity reconstruction
    png_recon = Path(OUT_DIR, f"bp_sample_{i:02d}_recon.pdf")
    plot_single(elem_maps[i], titles[i], png_recon, vmin=lo, vmax=hi)

    # voltage signal
    png_volt = Path(OUT_DIR, f"bp_sample_{i:02d}_volt.pdf")
    plot_voltage_signal(V_touch[i], titles[i], png_volt)

    print(f"Saved recon → {png_recon}, volt → {png_volt}")




# ===========================
# 7) No-touch recon + voltage (from real data)
# ===========================
N_NT_SAMPLES = 12  # how many no-touch exemplars to visualize

# Collect candidate no-touch rows (already computed earlier)
# mask_baseline: df["force_n"] < FORCE_BASELINE_THR
df_nt = df.loc[mask_baseline, :].copy()
if df_nt.empty:
    raise RuntimeError("No no-touch rows (force < threshold). Cannot render no-touch examples.")

# Use the same EIT columns (filtered non-constant, non-zero channels)
V_nt = df_nt[eit_cols].astype(np.float32).to_numpy()  # [K0, M]
K0 = V_nt.shape[0]

# Pick a few random no-touch rows for visualization
rng = np.random.default_rng(7)
sel = rng.choice(K0, size=min(N_NT_SAMPLES, K0), replace=False)

# Reconstruct no-touch maps
nt_elem_maps = []
nt_titles = []
for j, irow in enumerate(sel):
    v1_nt = V_nt[irow].astype(np.float32)

    # BP reconstruction (should be ~0-ish everywhere if v1_nt ~ v0)
    ds_node_nt = bp.solve(v1_nt, v0, normalize=True, log_scale=False).astype(np.float32)
    if ds_node_nt.shape[0] == mesh.node.shape[0] and ds_node_nt.shape[0] != mesh.element.shape[0]:
        ds_elem_nt = bp_node_to_element(ds_node_nt)
    elif ds_node_nt.shape[0] == mesh.element.shape[0]:
        ds_elem_nt = ds_node_nt
    else:
        raise RuntimeError(f"Unexpected BP length for no-touch: {ds_node_nt.shape[0]}")

    nt_elem_maps.append(ds_elem_nt)
    # Optional: include small force if present (it should be ~0)
    fN = df_nt.iloc[irow]["force_n"] if has_f else 0.0
    nt_titles.append(f"No-touch (F≈{fN:.2f} N)")

nt_elem_maps = np.stack(nt_elem_maps, axis=0)  # [N_NT_SAMPLES, n_elem]

# Use a tight, symmetric color scale (no-touch maps are near zero)
# Take a robust scale from the absolute values and make it symmetric
nt_hi = np.percentile(np.abs(nt_elem_maps), 99.0)
nt_lo, nt_hi = -nt_hi, nt_hi

# Save both the reconstruction and corresponding voltage for each no-touch exemplar
for j, irow in enumerate(sel):
    # Reconstruction
    png_recon_nt = Path(OUT_DIR, f"bp_notouch_{j:02d}_recon.pdf")
    plot_single(nt_elem_maps[j], nt_titles[j], png_recon_nt, vmin=nt_lo, vmax=nt_hi)

    # Voltage (use the exact no-touch frame used above)
    png_volt_nt = Path(OUT_DIR, f"bp_notouch_{j:02d}_volt.pdf")
    plot_voltage_signal(V_nt[irow], nt_titles[j], png_volt_nt)

    print(f"Saved no-touch recon → {png_recon_nt}, voltage → {png_volt_nt}")
