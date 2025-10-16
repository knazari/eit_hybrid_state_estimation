import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import re

plt.rcParams['pdf.fonttype'] = 42   # use TrueType fonts, not Type 3
plt.rcParams['ps.fonttype'] = 42



# Load dataset
df = pd.read_csv("data/touch_force_dataset.csv")

# Create output directory
os.makedirs("eda_outputs", exist_ok=True)

# Separate input and output
voltages = df.iloc[:, :-3]
labels = df.iloc[:, -3:]
labels.columns = ['x', 'y', 'force']

# 1. Voltage summary stats
voltages.describe().T.to_csv("eda_outputs/voltage_summary.csv")

# 2. Distribution plots of x, y, and force
plt.figure(figsize=(12, 4))
for i, col in enumerate(labels.columns):
    plt.subplot(1, 3, i + 1)
    sns.histplot(labels[col], kde=True, bins=30)
    plt.title(f"Distribution of {col}", fontsize=14)
    plt.xlabel(col, fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.tick_params(axis='both', which='major', labelsize=10)
plt.tight_layout()
plt.savefig("eda_outputs/label_distributions.pdf", dpi=300, bbox_inches="tight")
plt.close()

# 3. Scatter plot of touch locations, colored by force
plt.figure(figsize=(6, 6))
sc = plt.scatter(labels['x'], labels['y'], c=labels['force'], cmap='viridis', s=20)
cbar = plt.colorbar(sc)
cbar.set_label('Force', fontsize=12)
cbar.ax.tick_params(labelsize=10)
plt.title("Touch Locations Colored by Force", fontsize=14)
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.axis("equal")
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.savefig("eda_outputs/touch_locations_by_force.pdf", dpi=300, bbox_inches="tight")
plt.close()

# # 4. Correlation heatmap (voltage signals vs labels)
# combined = pd.concat([voltages, labels], axis=1)
# corr_matrix = combined.corr().loc[voltages.columns, ['x', 'y', 'force']]

# plt.figure(figsize=(10, 6))
# sns.heatmap(
#     corr_matrix, 
#     cmap='coolwarm', 
#     center=0, 
#     cbar_kws={"shrink": 0.99}
# )
# # plt.title("Correlation between Voltages and Output Labels", fontsize=16)
# # plt.xlabel("Output", fontsize=14)
# plt.ylabel("Voltage Index", fontsize=18, labelpad=2)
# plt.tick_params(axis='x', which='major', labelsize=18)
# plt.tick_params(axis='y', which='major', labelsize=14)
# plt.tight_layout()
# plt.savefig("eda_outputs/voltage_label_correlation.pdf", dpi=300, bbox_inches="tight")
# plt.close()

# Build correlation matrix as before
combined = pd.concat([voltages, labels], axis=1)
corr_matrix = combined.corr().loc[voltages.columns, ['x', 'y', 'force']]

# Helper to turn labels like "V0", "eit_2003" -> numeric index strings ("0", "2003")
def _label_to_index(s: str):
    m = re.search(r'\d+', str(s))
    return m.group(0) if m else str(s)

# Prepare first/last tick labels
first_lbl = _label_to_index(voltages.columns[0])
last_lbl  = _label_to_index(voltages.columns[-1])

plt.figure(figsize=(4.0, 2.5))  # smaller figure
ax = sns.heatmap(
    corr_matrix,
    cmap='coolwarm',
    center=0,
    cbar_kws={"shrink": 0.95, "pad": 0.03}
)

# y-axis: only first & last tick, rotated 90°
ax.set_yticks([0, len(voltages.columns) - 1])
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
plt.savefig(os.path.join("eda_outputs/voltage_label_correlation.pdf"), dpi=300, bbox_inches="tight")
plt.close()


# Save summary to CSV
summary_stats = combined.describe().T
summary_stats.to_csv("eda_outputs/summary_statistics.csv")
print(summary_stats)
