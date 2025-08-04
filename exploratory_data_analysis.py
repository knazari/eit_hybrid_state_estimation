import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

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
    plt.title(f"Distribution of {col}")
plt.tight_layout()
plt.savefig("eda_outputs/label_distributions.png")
plt.close()

# 3. Scatter plot of touch locations, colored by force
plt.figure(figsize=(6, 6))
sc = plt.scatter(labels['x'], labels['y'], c=labels['force'], cmap='viridis', s=20)
plt.colorbar(sc, label='Force')
plt.title("Touch Locations Colored by Force")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)
plt.savefig("eda_outputs/touch_locations_by_force.png")
plt.close()

# 4. Correlation heatmap (voltage signals vs labels)
combined = pd.concat([voltages, labels], axis=1)
corr_matrix = combined.corr().loc[voltages.columns, ['x', 'y', 'force']]

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, cmap='coolwarm', center=0)
plt.title("Correlation between Voltages and Output Labels")
plt.xlabel("Output")
plt.ylabel("Voltage Index")
plt.tight_layout()
plt.savefig("eda_outputs/voltage_label_correlation.png")
plt.close()

# 5. PCA to reduce voltage space to 2D and visualize
from sklearn.decomposition import PCA
scaler = StandardScaler()
voltages_scaled = scaler.fit_transform(voltages)
pca = PCA(n_components=2)
voltage_pca = pca.fit_transform(voltages_scaled)

plt.figure(figsize=(8, 6))
sc = plt.scatter(voltage_pca[:, 0], voltage_pca[:, 1], c=labels['force'], cmap='plasma', s=15)
plt.colorbar(sc, label='Force')
plt.title("PCA of Voltage Inputs Colored by Force")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.savefig("eda_outputs/pca_voltage_colored_by_force.png")
plt.close()

# Save summary to CSV
summary_stats = combined.describe().T
summary_stats.to_csv("eda_outputs/summary_statistics.csv")
print(summary_stats)
