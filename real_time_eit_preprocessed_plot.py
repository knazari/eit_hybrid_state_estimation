import serial, time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from sklearn.decomposition import PCA
from matplotlib.gridspec import GridSpec
import csv

# ---------------- Serial ----------------
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE   = 115200

# ---------------- Preproc / PCA settings ----------------
BASELINE_WARMUP_FRAMES = 20      # collect true "rest" first
USE_EMA_BASELINE       = False    # start False to reduce drift; set True after you see stability
EMA_ALPHA              = 0.005    # very slow if you enable EMA

PCA_WARMUP_FRAMES      = 30      # rest-only frames for PCA fit
N_COMPONENTS           = 6        # we’ll plot first 3
SCORE_SMOOTH_ALPHA     = 0.2      # EMA smoothing for displayed PC scores & recon error

HIST_LEN               = 600      # time history points to display (scores panel)

# ---------------- Helpers ----------------
def strip_prefix(raw: str) -> str:
    if raw.startswith("magnitudes:"):
        raw = raw[len("magnitudes:"):].strip()
    return raw

def parse_frame(raw: str) -> np.ndarray:
    raw = strip_prefix(raw)
    parts = [p for p in raw.split(',') if p.strip()!='']
    return np.array(list(map(float, parts)), dtype=np.float32) if parts else np.array([], np.float32)

def robust_frame_normalize(x: np.ndarray):
    """
    Remove per-frame global effects:
      1) subtract frame median (kills global offset)
      2) divide by robust scale (MAD) to remove global gain swings
    """
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    scale = mad if mad > 1e-6 else np.std(x) + 1e-6
    return (x - med) / scale

def zscore_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-8] = 1e-8
    return mu, sd

def zscore_apply(x, mu, sd):
    return (x - mu) / sd

def ema(prev, new, alpha):
    return (1-alpha)*prev + alpha*new

# ---------------- State ----------------
device = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=5)
time.sleep(2)
device.write(b"y")

baseline_ready = False
ema_baseline   = None
warmup_frames  = []

pca_ready = False
pca_warm  = []
pca       = None
z_mu = None
z_sd = None

# history for scores
hist_t   = deque(maxlen=HIST_LEN)
hist_pc1 = deque(maxlen=HIST_LEN)
hist_pc2 = deque(maxlen=HIST_LEN)
hist_pc3 = deque(maxlen=HIST_LEN)
hist_err = deque(maxlen=HIST_LEN)

# smoothed displays
pc1_s = pc2_s = pc3_s = err_s = 0.0

# ---------------- Plot layout ----------------
plt.ion()
fig = plt.figure(figsize=(12, 4.8))
gs = GridSpec(2, 1, height_ratios=[2.2, 1.0], hspace=0.35)

ax0 = fig.add_subplot(gs[0])
raw_line, = ax0.plot([], [], lw=1.0, label="raw")
det_line, = ax0.plot([], [], lw=1.2, label="detrended (raw - baseline)")
ax0.set_title("EIT Real-time Plot")
ax0.set_xlabel("Electrode Pair Index")
ax0.set_ylabel("Magnitude")
ax0.set_ylim(0.0, 1.0)
ax0.legend(loc="upper right")
txt = ax0.text(0.02, 0.95, "", transform=ax0.transAxes, va="top", ha="left")

ax1 = fig.add_subplot(gs[1])
pc1_line, = ax1.plot([], [], lw=1.2, label="PC1")
pc2_line, = ax1.plot([], [], lw=1.0, label="PC2")
pc3_line, = ax1.plot([], [], lw=1.0, label="PC3")
err_line, = ax1.plot([], [], lw=1.0, label="ReconErr")
ax1.set_xlabel("Frames")
ax1.set_ylabel("Scores / Err")
ax1.legend(loc="upper left")

frame_idx = 0
t0 = time.time()

try:
    while True:
        raw_str = device.readline().decode('utf-8', errors='ignore').strip()
        frame = parse_frame(raw_str)
        if frame.size == 0:
            continue
        

        # ---- baseline (static median first; optional slow EMA later) ----
        if ema_baseline is None:
            ema_baseline = frame.copy()

        if not baseline_ready:
            warmup_frames.append(frame)
            if len(warmup_frames) >= BASELINE_WARMUP_FRAMES:
                ema_baseline = np.median(np.stack(warmup_frames, axis=0), axis=0).astype(np.float32)
                baseline_ready = True
                print(f"[Baseline] ready using median of {len(warmup_frames)} frames")
        else:
            if USE_EMA_BASELINE:
                ema_baseline = ema(ema_baseline, frame, EMA_ALPHA)

        detrended = frame - ema_baseline


        # ---- per-frame normalization to remove global scale/offset ----
        #   This is key: makes PCs much more stable when nothing touches
        normed = robust_frame_normalize(detrended)

        # ---- PCA warmup & fit (on normed, AFTER baseline) ----
        if baseline_ready and not pca_ready:
            pca_warm.append(normed)
            if len(pca_warm) >= PCA_WARMUP_FRAMES:
                X = np.stack(pca_warm, axis=0)
                # channel z-score so each channel contributes equally
                z_mu, z_sd = zscore_fit(X)
                Xz = zscore_apply(X, z_mu, z_sd)
                pca = PCA(n_components=N_COMPONENTS, random_state=0)
                pca.fit(Xz)
                pca_ready = True
                print(f"[PCA] fitted on {len(X)} frames. EVR:", pca.explained_variance_ratio_)

        # ---- Online PCA metrics ----
        pc1=pc2=pc3=recon_err=np.nan
        if pca_ready:
            xz = zscore_apply(normed, z_mu, z_sd)
            scores = pca.transform(xz[None, :])[0]          # length N_COMPONENTS
            # reconstruction error from retained components
            xz_hat = pca.inverse_transform(scores[None, :])[0]
            recon_err = float(np.linalg.norm(xz - xz_hat))

            pc1, pc2, pc3 = float(scores[0]), float(scores[1]), float(scores[2]) if len(scores) > 2 else (float(scores[0]), float(scores[1]), 0.0)

            # smooth for display to reduce jitter
            pc1_s = ema(pc1_s, pc1, SCORE_SMOOTH_ALPHA)
            pc2_s = ema(pc2_s, pc2, SCORE_SMOOTH_ALPHA)
            pc3_s = ema(pc3_s, pc3, SCORE_SMOOTH_ALPHA)
            err_s = ema(err_s, recon_err, SCORE_SMOOTH_ALPHA)

            # history buffers
            hist_t.append(frame_idx)
            hist_pc1.append(pc1_s)
            hist_pc2.append(pc2_s)
            hist_pc3.append(pc3_s)
            hist_err.append(err_s)

        # ---- Update plots ----
        n = frame.size
        xs = np.arange(n)
        raw_line.set_data(xs, frame)
        det_line.set_data(xs, detrended)
        ax0.set_xlim(0, n)

        if pca_ready and len(hist_t) > 2:
            t = np.array(hist_t)
            pc1_line.set_data(t, np.array(hist_pc1))
            pc2_line.set_data(t, np.array(hist_pc2))
            pc3_line.set_data(t, np.array(hist_pc3))
            # err_line.set_data(t, np.array(hist_err))
            ax1.set_xlim(t.min(), t.max())
            # auto y for scores panel
            y = np.concatenate([np.array(hist_pc1), np.array(hist_pc2), np.array(hist_pc3)])#, np.array(hist_err)])
            lo, hi = np.percentile(y[~np.isnan(y)], [5, 95]) if np.isfinite(y).any() else ( -1, 1 )
            if hi > lo:
                pad = 0.2*(hi-lo)
                ax1.set_ylim(lo - pad, hi + pad)

            txt.set_text(f"PC1:{pc1_s:+.3f}\nPC2:{pc2_s:+.3f}\nPC3:{pc3_s:+.3f}\nReconErr:{err_s:.3f}")
        else:
            txt.set_text("Fitting PCA...")

        plt.draw()
        plt.pause(0.01)
        frame_idx += 1

except KeyboardInterrupt:
    print("\nInterrupted by user.")
finally:
    device.close()
    plt.ioff()
    plt.show()
