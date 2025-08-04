import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# === Step 1: Load dataset ===
df = pd.read_csv("/home/kiyanoush/eit-experiments/data/touch_force_dataset.csv")

# Separate features and targets
X = df.iloc[:, :-3].values  # 208 voltage features
y = df.iloc[:, -3:].values  # x, y, force

# === Step 2: Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Step 3: Feature scaling (important for SVR) ===
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# === Step 4: Train SVR using multi-output regressor ===
base_svr = SVR(kernel="rbf", C=100.0, epsilon=0.01)
model = MultiOutputRegressor(base_svr)
model.fit(X_train_scaled, y_train_scaled)

# === Step 5: Predict and inverse scale ===
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_unscaled = scaler_y.inverse_transform(y_test_scaled)

# === Step 6: Evaluation ===
mae = mean_absolute_error(y_test_unscaled, y_pred, multioutput="raw_values")
r2 = r2_score(y_test_unscaled, y_pred, multioutput="raw_values")

print("\n📊 SVR Performance (on Test Set):")
print(f"MAE - X:     {mae[0]:.4f}")
print(f"MAE - Y:     {mae[1]:.4f}")
print(f"MAE - Force: {mae[2]:.4f}")
print(f"R²  - X:     {r2[0]:.4f}")
print(f"R²  - Y:     {r2[1]:.4f}")
print(f"R²  - Force: {r2[2]:.4f}")

# === Step 7: Visualization ===
labels = ["x", "y", "force"]
for i in range(3):
    plt.figure(figsize=(4.5, 4))
    plt.scatter(y_test_unscaled[:, i], y_pred[:, i], alpha=0.7)
    plt.xlabel(f"True {labels[i]}")
    plt.ylabel(f"Predicted {labels[i]}")
    plt.title(f"SVR Prediction: {labels[i]}")
    plt.grid(True)
    plt.axis("equal")
    plt.plot([min(y_test_unscaled[:, i]), max(y_test_unscaled[:, i])],
             [min(y_test_unscaled[:, i]), max(y_test_unscaled[:, i])],
             'r--')  # y=x line
    plt.tight_layout()
    plt.savefig(f"data/svr_prediction_{labels[i]}.png", dpi=300)
    plt.close()

print("✅ SVR training and evaluation complete. Plots saved to data/")
