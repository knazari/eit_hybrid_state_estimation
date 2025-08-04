import matplotlib.pyplot as plt

# === R² values for x, y, and force — fill these with your results ===
models = [
    "SVR",
    "MLP",
    "AE-MLP",
    "CNN",
    "Transformer",
    "CNN+Transformer",
    "Physics-informed CNN"
]

r2_x = [0.9974, 0.9921, 0.9886, 0.9987, 0.9957, 0.9968, 0.9983]      # ← Replace with actual R² for x
r2_y = [0.9968, 0.9971, 0.9886, 0.9979, 0.9942, 0.9974, 0.9978]      # ← Replace with actual R² for y
r2_force = [0.8694, 0.9010, 0.8741, 0.9228, 0.9117, 0.9095, 0.9310]  # ← Replace with actual R² for force

# === Plot settings ===
x = range(len(models))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar([i - width for i in x], r2_x, width=width, label="x", alpha=0.8, color='r')
plt.bar(x, r2_y, width=width, label="y", alpha=0.8, color='b')
plt.bar([i + width for i in x], r2_force, width=width, label="force", alpha=0.8, color='g')

plt.xticks(x, models, rotation=25, ha='right')
plt.ylim(0.8, 1.05)
plt.ylabel("R² Score")
plt.title("Model Comparison – R² Scores")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig("/home/kiyanoush/eit-experiments/data/model_r2_comparison.png", dpi=300)
plt.show()
