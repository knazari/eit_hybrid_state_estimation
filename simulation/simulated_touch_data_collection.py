import numpy as np
import csv
import os
import pyeit.mesh as mesh
from pyeit.mesh import set_perm
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
from pyeit.eit.protocol import create as create_protocol
from pyeit.eit.fem import EITForward
import matplotlib.pyplot as plt

# === Create output directory if not exists ===
# os.makedirs("data", exist_ok=True)

# === Step 1: Create base mesh and protocol ===
n_el = 16
mesh_obj = mesh.create(n_el=n_el)
protocol = create_protocol(n_el=n_el, dist_exc=1, step_meas=1)
fwd = EITForward(mesh_obj, protocol)

# === Step 2: Simulate baseline voltages (no touch) ===
v_baseline = fwd.solve_eit(mesh_obj.perm)

# === Step 3: Define synthetic force-permittivity mapping ===
# n_force_levels = 20
# force_levels = np.linspace(0.0, 1.0, n_force_levels)
# perm_levels = 1.0 + 9.0 * force_levels  # Linear mapping: perm = 1 + 9*force

# === Step 3: Define synthetic force-permittivity mapping ===
n_force_samples = 100

# Option 1: Single normal distribution centered at 0.5
force_levels = np.clip(np.random.normal(loc=0.5, scale=0.2, size=n_force_samples), 0.0, 1.0)

# Option 2: Bimodal (mixture of two Gaussians centered at 0.4 and 0.6)
# n1 = n_force_samples // 2
# n2 = n_force_samples - n1
# f1 = np.random.normal(loc=0.4, scale=0.15, size=n1)
# f2 = np.random.normal(loc=0.6, scale=0.15, size=n2)
# force_levels = np.clip(np.concatenate([f1, f2]), 0.0, 1.0)

# Sort and remove close duplicates
force_levels = np.unique(np.round(force_levels, 3))

# Map force to permittivity (nonlinear mapping optional)
perm_levels = 1.0 + 9.0 * force_levels  # Still linear


# === Step 4: Define valid (x, y) touch points inside the mesh ===
# x_vals = np.linspace(-0.6, 0.6, 1)
# y_vals = np.linspace(-0.6, 0.6, 1)
touch_radius = 0.1

# rect_points = []
# for x in x_vals:
#     for y in y_vals:
#         if np.linalg.norm([x, y]) + touch_radius < 1.0:
#             rect_points.append((x, y))

n_radii = 8
n_angles = 24
r_vals = np.linspace(0.1, 1.0 - touch_radius, n_radii)
theta_vals = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)

polar_points = []
for r in r_vals:
    for theta in theta_vals:
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        polar_points.append((x, y))

# all_touch_points = list(set(rect_points + polar_points))
all_touch_points = polar_points
all_touch_points.sort()

# === Step 5: Write header for CSV file ===
out_file = "/home/kiyanoush/eit-experiments/data/touch_force_dataset.csv"
with open(out_file, "w", newline="") as f:
    writer = csv.writer(f)
    header = [f"v{i}" for i in range(len(v_baseline))] + ["x", "y", "force"]
    writer.writerow(header)

    # === Step 6: Loop through locations and force levels ===
    for (x, y) in all_touch_points:
        for force, perm in zip(force_levels, perm_levels):
            anomaly = [PyEITAnomaly_Circle(center=[x, y], r=touch_radius, perm=perm)]
            mesh_mod = set_perm(mesh_obj, anomaly=anomaly)
            v_touch = fwd.solve_eit(mesh_mod.perm)
            delta_v = v_touch - v_baseline

            row = list(delta_v) + [x, y, force]
            writer.writerow(row)
            print(f"Saved: (x={x:.2f}, y={y:.2f}, force={force:.2f}N, perm={perm:.1f})")

print(f"\n✅ Dataset saved to: {out_file}")

# === Plotting mesh and touch points ===
pts = mesh_obj.node
el_pos = mesh_obj.el_pos
touch_points = np.array(all_touch_points)

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_title("Synthetic Touch Locations on EIT Sensor")
ax.set_aspect("equal")
ax.set_xlim(-1.1, 1.1)
ax.set_ylim(-1.1, 1.1)
ax.plot(touch_points[:, 0], touch_points[:, 1], 'bo', label="Touch locations")
ax.plot(pts[el_pos, 0], pts[el_pos, 1], 'ro', label="Electrodes")
circle = plt.Circle((0, 0), 1.0, color='gray', fill=False, linestyle='--')
ax.add_patch(circle)
ax.legend()
plt.savefig("/home/kiyanoush/eit-experiments/data/touch_locations_plot.png", dpi=300)
plt.close()
print("✅ Touch location map saved as 'data/touch_locations_plot.png'")
