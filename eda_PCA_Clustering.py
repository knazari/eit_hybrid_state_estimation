# Re-run after kernel reset: full PCA + clustering visualization pipeline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import os

# Load data
df = pd.read_csv("data/touch_force_dataset.csv")
X_voltage = df.iloc[:, :-3].values
y = df.iloc[:, -3:].values
x_pos = y[:, 0]
y_pos = y[:, 1]
force = y[:, 2]

# Scale the voltage data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_voltage)

# PCA reduction to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Create output directory
os.makedirs("eda_outputs", exist_ok=True)

# === Plot 1: PCA colored by X location ===
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=x_pos, cmap='coolwarm', s=30)
plt.colorbar(label='X Position')
plt.title("PCA of Voltage Inputs Colored by X Position")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("eda_outputs/pca_voltage_colored_by_x.png")
plt.close()

# === Plot 2: PCA colored by Y location ===
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_pos, cmap='coolwarm', s=30)
plt.colorbar(label='Y Position')
plt.title("PCA of Voltage Inputs Colored by Y Position")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("eda_outputs/pca_voltage_colored_by_y.png")
plt.close()

# === Clustering: KMeans ===
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(X_pca)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='tab10', s=30)
plt.title("KMeans Clustering on PCA-Reduced Voltage Data")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("eda_outputs/pca_kmeans_clusters_2.png")
plt.close()

# === Clustering: DBSCAN ===
dbscan = DBSCAN(eps=0.003, min_samples=10)
dbscan_labels = dbscan.fit_predict(X_pca)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels, cmap='tab20', s=30)
plt.title("DBSCAN Clustering on PCA-Reduced Voltage Data")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig("eda_outputs/pca_dbscan_clusters.png")
plt.close()

