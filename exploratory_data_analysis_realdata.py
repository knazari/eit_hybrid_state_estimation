import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import re

plt.rcParams['pdf.fonttype'] = 42   # use TrueType fonts, not Type 3
plt.rcParams['ps.fonttype'] = 42


# ----------------------------
# Paths
# ----------------------------
DATA_PATH = "data/merged_with_probe.csv"
OUT_DIR   = "eda_outputs_real"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv(DATA_PATH)

# ----------------------------
# Select EIT channels (exclude 't_eit' if present) and drop near-constant ones
# ----------------------------
eit_cols = [c for c in df.columns if c.startswith("eit_") and c.lower() != "t_eit"]
if len(eit_cols) == 0:
    raise RuntimeError("No EIT columns found (eit_*). Check DATA_PATH / headers.")

eps = 1e-9
variable_eit_cols = []
for c in eit_cols:
    std_c = df[c].std(skipna=True)
    if df[c].nunique(dropna=False) > 1 and std_c is not None and std_c > eps:
        variable_eit_cols.append(c)

if len(variable_eit_cols) == 0:
    raise RuntimeError("All EIT channels are constant/near-constant after filtering.")

# ----------------------------
# Targets & masks
# ----------------------------
need_cols = ["contact_u_m", "contact_v_m", "force_n"]
for c in need_cols:
    if c not in df.columns:
        raise RuntimeError(f"Missing column '{c}' in CSV.")

targets = df[need_cols].copy()

mask_contact  = targets["force_n"].notna() & (targets["force_n"] >= 0.5)
mask_baseline = targets["force_n"].notna() & (targets["force_n"] < 0.05)

if mask_baseline.sum() == 0:
    raise RuntimeError("No 'no-touch' rows (force_n < 0.05) found to compute baseline.")

# ----------------------------
# Compute baseline and build features/labels
# ----------------------------
v_ref = df.loc[mask_baseline, variable_eit_cols].astype(np.float32).to_numpy()
v_ref = np.nanmean(v_ref, axis=0, dtype=np.float64).astype(np.float32)  # [D]

X_raw = df.loc[mask_contact, variable_eit_cols].astype(np.float32).to_numpy()
X = (X_raw - v_ref[None, :]).astype(np.float32)

labels = targets.loc[mask_contact, :].copy()
labels.columns = ['x', 'y', 'force']  # rename for consistency

# --- Rename EIT channels to V0..Vn for consistency with sim dataset ---
voltage_labels = [f"V{i}" for i in range(X.shape[1])]
voltages = pd.DataFrame(X, columns=voltage_labels)

print(f"EIT channels total: {len(eit_cols)} | kept: {len(variable_eit_cols)} | removed: {len(eit_cols) - len(variable_eit_cols)}")
print(f"Samples total: {len(df)} | kept (contact): {mask_contact.sum()} | baseline rows: {mask_baseline.sum()}")

# ----------------------------
# 1) Voltage summary stats
# ----------------------------
voltages.describe().T.to_csv(os.path.join(OUT_DIR, "voltage_summary.csv"))

# ----------------------------
# 2) Distributions of x, y, force
# ----------------------------
import seaborn as sns
plt.figure(figsize=(12, 4))
for i, col in enumerate(['x', 'y', 'force']):
    plt.subplot(1, 3, i + 1)
    sns.histplot(labels[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}", fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "label_distributions.pdf"), dpi=300, bbox_inches="tight")
plt.close()

# ----------------------------
# 3) Scatter of touch locations colored by force
# ----------------------------
plt.figure(figsize=(6.5, 6))
sc = plt.scatter(labels['x'], labels['y'], c=labels['force'], cmap='Reds', s=20)
cbar = plt.colorbar(sc)
cbar.set_label('Force (N)', fontsize=16)
cbar.ax.tick_params(labelsize=10)
# plt.title("Touch Locations Colored by Force", fontsize=14)
plt.xlabel("x [m]", fontsize=16)
plt.ylabel("y [m]", fontsize=16)
plt.axis("equal")
# plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.savefig(os.path.join(OUT_DIR, "touch_locations_by_force.pdf"), dpi=300)
plt.close()

# ----------------------------
# 4) Correlation heatmap (voltages vs labels)
# ----------------------------
# combined = pd.concat([voltages, labels], axis=1)
# corr_matrix = combined.corr().loc[voltage_labels, ['x', 'y', 'force']]

# plt.figure(figsize=(10, 6))
# sns.heatmap(
#     corr_matrix,
#     cmap='coolwarm',
#     center=0,
#     cbar_kws={"shrink": 0.99}
# )
# plt.ylabel("Voltage Index", fontsize=18, labelpad=2)
# plt.tick_params(axis='x', which='major', labelsize=18)
# plt.tick_params(axis='y', which='major', labelsize=14)
# plt.tight_layout()
# plt.savefig(os.path.join(OUT_DIR, "voltage_label_correlation.pdf"), dpi=300, bbox_inches="tight")
# plt.close()


# Build correlation matrix as before
combined = pd.concat([voltages, labels], axis=1)
corr_matrix = combined.corr().loc[voltage_labels, ['x', 'y', 'force']]

# Helper to turn labels like "V0", "eit_2003" -> numeric index strings ("0", "2003")
def _label_to_index(s: str):
    m = re.search(r'\d+', str(s))
    return m.group(0) if m else str(s)

# Prepare first/last tick labels
first_lbl = _label_to_index(voltage_labels[0])
last_lbl  = _label_to_index(voltage_labels[-1])

plt.figure(figsize=(4.0, 2.5))  # smaller figure
ax = sns.heatmap(
    corr_matrix,
    cmap='coolwarm',
    center=0,
    cbar_kws={"shrink": 0.95, "pad": 0.03}
)

# y-axis: only first & last tick, rotated 90°
ax.set_yticks([0, len(voltage_labels) - 1])
ax.set_yticklabels([first_lbl, last_lbl], rotation=90, va='center')
ax.set_ylabel("Voltage Index", fontsize=16, labelpad=-1)

# x-axis ticks (only three labels here); rotate a bit if you want to save space
ax.tick_params(axis='x', labelsize=18, rotation=0)
ax.tick_params(axis='y', labelsize=18)

# Get colorbar from the heatmap
cbar = ax.collections[0].colorbar

# Get vmin and vmax from the mappable (the first collection in ax)
vmin, vmax = ax.collections[0].get_clim()

# Only keep first and last ticks
cbar.set_ticks([vmin, vmax])

# Optional: format labels nicely
cbar.set_ticklabels([f"{vmin:.2f}", f"{vmax:.2f}"])

# Rotate tick labels fully vertical
cbar.ax.tick_params(labelsize=16)
for tick_label in cbar.ax.get_yticklabels():
    tick_label.set_rotation(90)
    tick_label.set_va('center')


plt.tight_layout(pad=0.1)
plt.savefig(os.path.join(OUT_DIR, "voltage_label_correlation.pdf"), dpi=300, bbox_inches="tight")
plt.close()

# ----------------------------
# 5) Save summary to CSV
# ----------------------------
summary_stats = combined.describe().T
summary_stats.to_csv(os.path.join(OUT_DIR, "summary_statistics.csv"))
print(summary_stats)
