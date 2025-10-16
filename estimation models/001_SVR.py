import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# ===========================
# Config: Choose dataset
# ===========================
DATA_SOURCE = "sim"   # "sim" or "real"

if DATA_SOURCE == "sim":
    DATA_PATH = "/home/kiyanoush/Projects/eit_hybrid_state_estimation/data/touch_force_dataset_sim2real.csv"
    OUT_DIR = "results_sim"
elif DATA_SOURCE == "real":
    DATA_PATH = "data/merged_with_probe.csv"  # change to your real csv
    OUT_DIR = "results_real"
else:
    raise ValueError("DATA_SOURCE must be 'sim' or 'real'")

os.makedirs(OUT_DIR, exist_ok=True)

# ===========================
# Step 1: Load and split
# ===========================
df = pd.read_csv(DATA_PATH)

if DATA_SOURCE == "sim":
    # Simulated: last 3 columns = x, y, force
    X = df.iloc[:, :-3].values
    y = df.iloc[:, -3:].values
    label_names = ["x", "y", "force"]
else:
    # 1) Pick EIT columns (exclude 't_eit' if present)
    eit_cols = [c for c in df.columns if c.startswith("eit_") and c.lower() != "t_eit"]
    if len(eit_cols) == 0:
        raise RuntimeError("No EIT columns found (eit_0, eit_1, ...). Did you point DATA_PATH to a real CSV?")

    # 2) Remove EIT channels that are constant / near-constant
    eps = 1e-9
    variable_eit_cols = []
    for c in eit_cols:
        std_c = df[c].std(skipna=True)
        if df[c].nunique(dropna=False) > 1 and std_c is not None and std_c > eps:
            variable_eit_cols.append(c)

    if len(variable_eit_cols) == 0:
        raise RuntimeError("All EIT channels are constant/near-constant after filtering. Check the data.")

    # 3) Build targets
    targets = df[["contact_u_m", "contact_v_m", "force_n"]].copy()

    # Masks
    mask_contact   = targets["force_n"].notna() & (targets["force_n"] >= 0.5)     # keep for training
    mask_baseline  = targets["force_n"].notna() & (targets["force_n"] <  0.05)     # use to compute baseline

    # 4) Compute baseline vector v_ref over baseline rows (per-channel mean)
    if mask_baseline.sum() == 0:
        raise RuntimeError("No 'no-touch' rows (force_n < 0.5) found to compute baseline.")
    v_ref = df.loc[mask_baseline, variable_eit_cols].astype(np.float32).to_numpy()
    # nanmean guards against any stray NaNs
    v_ref = np.nanmean(v_ref, axis=0, dtype=np.float64).astype(np.float32)  # shape: [D]

    # 5) Apply row filter (contact cases), subtract baseline from EIT signals
    X_raw = df.loc[mask_contact, variable_eit_cols].astype(np.float32).to_numpy()  # [N, D]
    X = (X_raw - v_ref[None, :]).astype(np.float32)                                # baseline-subtracted features

    y = targets.loc[mask_contact, :].to_numpy(dtype=np.float32)
    label_names = ["x", "y", "force"]

    # (Optional) Keep v_ref for later use (e.g., saving with model to normalize test-time inputs)
    baseline_vector = v_ref  # np.ndarray shape [D]
    
    # (Optional) quick diagnostics
    print(f"EIT channels total: {len(eit_cols)} | kept: {len(variable_eit_cols)} | removed: {len(eit_cols) - len(variable_eit_cols)}")
    print(f"Samples total: {len(df)} | kept: {mask_contact.sum()} | dropped: {(~mask_contact).sum()}")


print(f"Dataset: {DATA_SOURCE} | Samples: {X.shape[0]} | Features: {X.shape[1]}")

# ===========================
# Step 2: Train-test split
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===========================
# Step 3: Feature scaling
# ===========================
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

y_train_scaled = scaler_y.fit_transform(y_train)
y_test_scaled = scaler_y.transform(y_test)

# ===========================
# Step 4: Train SVR
# ===========================
base_svr = SVR(kernel="rbf", C=100.0, epsilon=0.01)
model = MultiOutputRegressor(base_svr)
model.fit(X_train_scaled, y_train_scaled)

# ===========================
# Step 5: Predict
# ===========================
y_pred_scaled = model.predict(X_test_scaled)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_unscaled = scaler_y.inverse_transform(y_test_scaled)

# ===========================
# Step 6: Evaluation
# ===========================
mae = mean_absolute_error(y_test_unscaled, y_pred, multioutput="raw_values")
r2 = r2_score(y_test_unscaled, y_pred, multioutput="raw_values")

print("\n📊 SVR Performance (on Test Set):")
for i, lbl in enumerate(label_names):
    print(f"MAE - {lbl:5s}: {mae[i]:.4f}")
for i, lbl in enumerate(label_names):
    print(f"R²  - {lbl:5s}: {r2[i]:.4f}")

# ===========================
# Step 7: Visualization
# ===========================
for i, lbl in enumerate(label_names):
    plt.figure(figsize=(4.5, 4))
    plt.scatter(y_test_unscaled[:, i], y_pred[:, i], alpha=0.7)
    plt.xlabel(f"True {lbl}")
    plt.ylabel(f"Predicted {lbl}")
    plt.title(f"SVR Prediction: {lbl}")
    plt.grid(True)
    plt.axis("equal")
    min_val, max_val = min(y_test_unscaled[:, i]), max(y_test_unscaled[:, i])
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"svr_prediction_{lbl}.png"), dpi=300)
    plt.close()

print(f"✅ SVR training complete. Results saved in {OUT_DIR}/")
