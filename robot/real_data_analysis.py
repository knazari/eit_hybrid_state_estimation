import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# =====================
# Config
# =====================
DATA_PATH = "/home/kiyanoush/Projects/eit_hybrid_state_estimation/data/merged_with_probe.csv"  # <-- change to one of your collected CSVs
OUT_DIR = "eda_outputs_real"
os.makedirs(OUT_DIR, exist_ok=True)

# =====================
# Load data
# =====================
df = pd.read_csv(DATA_PATH)

# Identify EIT columns (they all start with "eit_")
eit_cols = [c for c in df.columns if c.startswith("eit_")]
meta_cols = [c for c in df.columns if c not in eit_cols]

# Labels of interest
labels = df[["contact_u_m", "contact_v_m", "force_n", "depth_m"]].copy()
labels.columns = ["x", "y", "force", "depth"]

# Voltages = EIT
voltages = df[eit_cols].copy()

# =====================
# 1. Voltage summary stats
# =====================
voltages.describe().T.to_csv(os.path.join(OUT_DIR, "voltage_summary.csv"))

# =====================
# 2. Distribution plots of x, y, force, depth
# =====================
plt.figure(figsize=(16, 4))
for i, col in enumerate(labels.columns):
    plt.subplot(1, 4, i + 1)
    sns.histplot(labels[col].dropna(), kde=True, bins=30)
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "label_distributions.png"))
plt.close()

# =====================
# 3. Scatter plot of touch locations, colored by force
# =====================
plt.figure(figsize=(6, 6))
sc = plt.scatter(labels["x"], labels["y"], c=labels["force"], cmap="viridis", s=20)
plt.colorbar(sc, label="Force [N]")
plt.title("Touch Locations Colored by Force")
plt.xlabel("u [m]")
plt.ylabel("v [m]")
plt.axis("equal")
plt.grid(True)
plt.savefig(os.path.join(OUT_DIR, "touch_locations_by_force.png"))
plt.close()

# =====================
# 4. Scatter plot of touch locations, colored by depth
# =====================
plt.figure(figsize=(6, 6))
sc = plt.scatter(labels["x"], labels["y"], c=labels["depth"], cmap="plasma", s=20)
plt.colorbar(sc, label="Depth [m]")
plt.title("Touch Locations Colored by Depth")
plt.xlabel("u [m]")
plt.ylabel("v [m]")
plt.axis("equal")
plt.grid(True)
plt.savefig(os.path.join(OUT_DIR, "touch_locations_by_depth.png"))
plt.close()

# =====================
# 5. Correlation heatmap (EIT signals vs labels)
# =====================
combined = pd.concat([voltages, labels], axis=1)
corr_matrix = combined.corr().loc[voltages.columns, ["x", "y", "force", "depth"]]

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
plt.title("Correlation between EIT Channels and Labels")
plt.xlabel("Output")
plt.ylabel("EIT Channel Index")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "voltage_label_correlation.png"))
plt.close()

# =====================
# 6. PCA of EIT voltages, color by force
# =====================
scaler = StandardScaler()
voltages_scaled = scaler.fit_transform(voltages.fillna(0))
pca = PCA(n_components=2)
voltage_pca = pca.fit_transform(voltages_scaled)

plt.figure(figsize=(8, 6))
sc = plt.scatter(voltage_pca[:, 0], voltage_pca[:, 1], c=labels["force"], cmap="plasma", s=15)
plt.colorbar(sc, label="Force [N]")
plt.title("PCA of EIT Voltages Colored by Force")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.savefig(os.path.join(OUT_DIR, "pca_voltage_colored_by_force.png"))
plt.close()

# =====================
# 7. Save summary statistics
# =====================
summary_stats = combined.describe().T
summary_stats.to_csv(os.path.join(OUT_DIR, "summary_statistics.csv"))
print(summary_stats.head())
