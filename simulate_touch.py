import numpy as np
import matplotlib.pyplot as plt
import pyeit.mesh as mesh
from pyeit.mesh import set_perm
from pyeit.mesh.wrapper import PyEITAnomaly_Circle
from pyeit.eit.protocol import create as create_protocol
from pyeit.eit.fem import EITForward
from pyeit.eit.bp import BP  # Back-projection solver
from pyeit.eit.jac import JAC

# Step 1: Create mesh
n_el = 16
mesh_obj = mesh.create(n_el=n_el)

# Step 2: Add anomaly to simulate touch
anomaly = [PyEITAnomaly_Circle(center=[0.5, 0.5], r=0.1, perm=10.0)]
mesh_new = set_perm(mesh_obj, anomaly=anomaly)

# Step 3: Extract node, element
pts = mesh_obj.node
tri = mesh_obj.element

# Step 4: Plot delta conductivity (anomaly vs baseline)
fig, axes = plt.subplots(2, 1, constrained_layout=True, figsize=(6, 9))

# Top plot: input anomaly (delta permittivity)
ax = axes[0]
ax.axis("equal")
ax.set_title(r"Input $\Delta$ Conductivities")
delta_perm = np.real(mesh_new.perm - mesh_obj.perm)
im = ax.tripcolor(pts[:, 0], pts[:, 1], tri, delta_perm, shading="flat")
ax.plot(pts[mesh_obj.el_pos, 0], pts[mesh_obj.el_pos, 1], 'ro')

# Bottom plot: placeholder for reconstructed result
ax1 = axes[1]
ax1.tripcolor(pts[:, 0], pts[:, 1], tri, np.zeros_like(delta_perm), shading="flat")
ax1.set_title(r"Reconstituted $\Delta$ Conductivities (Placeholder)")
ax1.axis("equal")

# Colorbar
fig.colorbar(im, ax=axes.ravel().tolist())
# plt.show()


# Step 1: Define a stimulation protocol (adjacent current injection)
protocol = create_protocol(n_el=n_el, dist_exc=1, step_meas=1)

# Step 2: Initialize the forward solver
fwd = EITForward(mesh_obj, protocol)

# Step 3: Solve voltage measurements
v_baseline = fwd.solve_eit(mesh_obj.perm)      # no anomaly
v_anomaly = fwd.solve_eit(mesh_new.perm)       # with anomaly

# Step 4: Calculate differential voltages
delta_v = v_anomaly - v_baseline

# print("Baseline voltages:\n", v_baseline)
# print("Anomaly voltages:\n", v_anomaly)
# print("Delta voltages:\n", delta_v)

# # Save for training if needed
# np.savetxt("/home/kiyanoush/eit-experiments/data/voltages_baseline.csv", v_baseline, delimiter=",")
# np.savetxt("/home/kiyanoush/eit-experiments/data/voltages_anomaly.csv", v_anomaly, delimiter=",")
# np.savetxt("/home/kiyanoush/eit-experiments/data/voltages_delta.csv", delta_v, delimiter=",")

# Step 1: Initialize the solver
eit = BP(mesh_obj, protocol)
# Step 2: Run setup (this computes the projection matrix)
eit.setup()
# Step 3: Reconstruct the conductivity difference
ds = eit.solve(v_anomaly, v_baseline, normalize=True)  # delta_sigma

# Bottom plot: reconstructed delta conductivity
ax1 = axes[1]
im = ax1.tripcolor(pts[:, 0], pts[:, 1], tri, ds, shading="flat", cmap=plt.cm.viridis)
ax1.set_title(r"Reconstructed $\Delta$ Conductivities (BP)")
ax1.axis("equal")

fig.colorbar(im, ax=axes.ravel().tolist())

# plt.show()

# Step 1: Initialize the JAC solver
eit = JAC(mesh_obj, protocol)
# Step 2: Configure solver parameters (these matter!)
eit.setup(p=0.5, lamb=0.01)  # p = regularization norm, lamb = damping factor
# Step 3: Reconstruct the conductivity change
ds_jac = eit.solve(v_anomaly, v_baseline, normalize=True)

# Create a new figure to compare JAC
fig2, ax2 = plt.subplots()
im2 = ax2.tripcolor(pts[:, 0], pts[:, 1], tri, ds_jac, shading="flat", cmap=plt.cm.viridis)
ax2.set_title(r"Reconstructed $\Delta$ Conductivities (JAC)")
ax2.axis("equal")
fig2.colorbar(im2)
plt.show()

