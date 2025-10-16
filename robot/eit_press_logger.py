#!/usr/bin/env python3
import time
import json
import threading
import queue
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import URBasic
try:
    import serial
except ImportError:
    serial = None

# =====================
# CONFIG — EDIT ME
# =====================
CONFIG = {
    # Robot
    "ROBOT_IP": "169.254.76.5",
    # Sensor center (approach plane) pose [x,y,z,rx,ry,rz]
    "SENSOR_CENTER_POSE": [-0.593, -0.201, -0.092, 0.0, 3.142, 0.0],
    # "SENSOR_CENTER_POSE": [-0.591, -0.201, -0.072, 0.0, 3.142, 0.0],

    # Geometry & pattern (matches simulator)
    "SENSOR_RADIUS_M": 0.088,      # 9.4 cm radius in meters
    "N_RADII": 8,
    "N_ANGLES": 24,
    "TOUCH_RADIUS_FRAC": 0.14,     # probe_radius / sensor_radius = 13mm / 94mm ≈ 0.138

    # Motion / stepping-to-target-force
    "STEP_MM": 1.0,                # step size toward the surface (mm)
    "MAX_EXTRA_DEPTH_MM": 15.0,    # safety cap on total additional depth (mm)
    "SETTLE_S": 3.00,              # settle time after each step (s)
    "TARGET_FORCE_N": 8.0,         # target contact force (N)
    "MAX_FORCE_N": 20.0,           # hard safety limit (N)

    # Speeds/accels
    "MOVE_V": 0.05,
    "MOVE_A": 0.05,
    "PRESS_V": 0.01,
    "PRESS_A": 0.01,

    # Serial ports
    "FORCE_PORT": "/dev/ttyACM0",
    "FORCE_BAUD": 115200,
    "EIT_PORT": "/dev/ttyACM1",
    "EIT_BAUD": 115200,

    # Output
    "OUT_DIR": "./data",
    "SESSION_TAG": "press_polar_steps",

    # Live logging
    "CSV_LIVE": True,          # append each sample immediately to CSV
    "PARQUET_EVERY_N": 0,      # >0 to also write parquet chunks every N rows (optional)
}

# =====================
# Simple helpers (robot)
# =====================

def connect_robot(ip: str):
    model = URBasic.robotModel.RobotModel()
    robot = URBasic.urScriptExt.UrScriptExt(host=ip, robotModel=model)
    try:
        robot.reset_error()
        time.sleep(0.5)
    except Exception:
        pass
    return robot

def movel(robot, pose, a=0.05, v=0.05):
    robot.movel(pose, a=a, v=v)

def approach_pose(center_pose, du_m, dv_m):
    x, y, z, rx, ry, rz = center_pose
    return [x + du_m, y + dv_m, z, rx, ry, rz]

def set_depth(pose, depth_m):
    p = pose.copy()
    p[2] = p[2] - depth_m
    return p

# =====================
# Serial readers (minimal)
# =====================
class SerialLineReader(threading.Thread):
    """Generic serial line reader with optional init function."""
    def __init__(self, port, baud, parse_fn, out_q, name, init_fn=None):
        super().__init__(daemon=True, name=name)
        if serial is None:
            raise RuntimeError("pyserial not installed. pip install pyserial")
        self.ser = serial.Serial(port, baud, timeout=0.05)
        self.parse_fn = parse_fn
        self.q = out_q
        self.stop_evt = threading.Event()
        self.init_fn = init_fn

    def run(self):
        # Optional port init (e.g., send 'y' trigger; discard baseline)
        if self.init_fn is not None:
            try:
                self.init_fn(self.ser)
            except Exception as e:
                print(f"[{self.name}] init_fn error: {e}")

        while not self.stop_evt.is_set():
            line = self.ser.readline()
            if not line:
                continue
            t_now = time.monotonic()
            sample = self.parse_fn(line, t_now)
            if sample is not None:
                self.q.put(sample)

    def stop(self):
        self.stop_evt.set()
        try:
            self.ser.close()
        except Exception:
            pass

# ---- Parsers

def parse_force(line: bytes, t_now: float):
    s = line.decode(errors='ignore').strip()
    if not s:
        return None
    try:
        return {"t": t_now, "force_n": float(s)}
    except Exception:
        return None

def parse_eit(line: bytes, t_now: float):
    """Parse EIT CSV lines; strip 'magnitudes:' prefix if present."""
    s = line.decode(errors='ignore').strip()
    if not s:
        return None
    if s.startswith("magnitudes:"):
        s = s[len("magnitudes:"):].strip()
    # JSON path not needed for your board; keep CSV
    try:
        vals = [float(x) for x in s.split(',') if x.strip() != ""]
        if len(vals) == 0:
            return None
        return {"t": t_now, "readings": vals}
    except Exception:
        return None

def init_eit_port(ser):
    """Send trigger 'y', discard first baseline line (with optional prefix)."""
    time.sleep(0.2)
    ser.write(b"y")
    ser.flush()
    # Read & discard baseline line
    _ = ser.readline()

# =====================
# Contact pattern (same as simulator)
# =====================

def generate_polar_points(n_radii, n_angles, touch_radius_frac):
    r_vals = np.linspace(0.1, 1.0 - touch_radius_frac, n_radii)
    theta_vals = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    pts = []
    for r in r_vals:
        for th in theta_vals:
            x = r * np.cos(th)
            y = r * np.sin(th)
            pts.append((float(x), float(y), float(r), float(th)))
    pts.sort(key=lambda p: (round(p[2], 6), round(p[3], 6)))
    return pts  # list of (x_norm, y_norm, r, theta)

# =====================
# Sampling utilities (no averaging)
# =====================

def sample_force_now(q_force):
    """Return the most recent force value available right now (no averaging)."""
    last = None
    while True:
        try:
            last = q_force.get_nowait()
        except queue.Empty:
            break
    if last is None:
        return None
    return float(last.get("force_n")) if "force_n" in last else None

def sample_eit_now(q_eit):
    """Return the most recent EIT sample (no averaging)."""
    last = None
    while True:
        try:
            last = q_eit.get_nowait()
        except queue.Empty:
            break
    return last  # dict with keys: t, readings

# =====================
# Main
# =====================

def main():
    cfg = CONFIG
    Path(cfg["OUT_DIR"]).mkdir(parents=True, exist_ok=True)

    # === Run file paths & live writers ===
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_base = Path(cfg["OUT_DIR"]) / f"{cfg['SESSION_TAG']}_{ts}"
    csv_path = run_base.with_suffix('.csv')

    # We will build a stable header including EIT columns once we know channel count
    header_cols = [
        "t", "tcp_x", "tcp_y", "tcp_z", "tcp_rx", "tcp_ry", "tcp_rz",
        "contact_idx", "contact_u_m", "contact_v_m", "contact_r_norm",
        "contact_theta", "depth_m", "force_n", "t_eit"
    ]
    eit_channel_count = 0  # will set after first sample

    def write_row_csv(rec: dict):
        """Append one row to CSV immediately with a stable header including eit_* columns."""
        nonlocal header_cols, eit_channel_count
        if not cfg.get("CSV_LIVE", True):
            return
        # Ensure all expected columns exist in rec (fill missing with None)
        if eit_channel_count > 0:
            # make sure eit_0..eit_{k-1} keys exist
            for i in range(eit_channel_count):
                rec.setdefault(f"eit_{i}", None)
        df1 = pd.DataFrame([rec], columns=header_cols + [f"eit_{i}" for i in range(eit_channel_count)])
        df1.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)

    parquet_buf = []
    parquet_chunk_id = 0

    def maybe_write_parquet_chunk(rec: dict):
        """Write small parquet files every N rows if enabled."""
        nonlocal parquet_buf, parquet_chunk_id
        N = int(cfg.get("PARQUET_EVERY_N", 0))
        if N <= 0:
            return
        parquet_buf.append(rec)
        if len(parquet_buf) >= N:
            chunk_dir = Path(cfg["OUT_DIR"]) / f"{cfg['SESSION_TAG']}_{ts}_chunks"
            chunk_dir.mkdir(parents=True, exist_ok=True)
            chunk_path = chunk_dir / f"part_{parquet_chunk_id:05d}.parquet"
            pd.DataFrame(parquet_buf).to_parquet(chunk_path, index=False)
            parquet_buf.clear()
            parquet_chunk_id += 1

    # --- Robot ---
    robot = connect_robot(cfg["ROBOT_IP"])
    movel(robot, cfg["SENSOR_CENTER_POSE"], a=cfg["MOVE_A"], v=cfg["MOVE_V"])

    # --- Serial threads ---
    q_force = queue.Queue(maxsize=4000)
    q_eit = queue.Queue(maxsize=4000)
    force_reader = SerialLineReader(cfg["FORCE_PORT"], cfg["FORCE_BAUD"], parse_force, q_force, "force")
    eit_reader = SerialLineReader(cfg["EIT_PORT"], cfg["EIT_BAUD"], parse_eit, q_eit, "eit", init_fn=init_eit_port)
    force_reader.start(); eit_reader.start()

    # ---- Determine EIT channel count before starting motions (to fix CSV header)
    t0 = time.time()
    while True:
        try:
            samp = q_eit.get(timeout=2.0)  # wait up to 2s for first EIT frame
            if samp and "readings" in samp and len(samp["readings"]) > 0:
                eit_channel_count = len(samp["readings"])
                break
        except queue.Empty:
            pass
        if time.time() - t0 > 5.0:
            print("[warn] No EIT data seen within 5s; proceeding without fixed EIT columns.")
            break

    # If we learned channel count, extend header now
    if eit_channel_count > 0:
        header_cols = header_cols + [f"eit_{i}" for i in range(eit_channel_count)]

    # --- Contact plan ---
    polar_pts = generate_polar_points(cfg["N_RADII"], cfg["N_ANGLES"], cfg["TOUCH_RADIUS_FRAC"])
    plan = [
        {
            "idx": i,
            "r": r,
            "theta": th,
            "u_m": x * cfg["SENSOR_RADIUS_M"],
            "v_m": y * cfg["SENSOR_RADIUS_M"],
        }
        for i, (x, y, r, th) in enumerate(polar_pts)
    ]

    # --- Logging buffers ---
    rows = []
    step_m = cfg["STEP_MM"] / 1000.0
    max_depth_m = cfg["MAX_EXTRA_DEPTH_MM"] / 1000.0

    try:
        for pt in plan:
            # 1) Approach in plane (safe height, no contact)
            appr = approach_pose(cfg["SENSOR_CENTER_POSE"], pt["u_m"], pt["v_m"])
            movel(robot, appr, a=cfg["MOVE_A"], v=cfg["MOVE_V"])
            time.sleep(0.2)

            # 2) Step down until target force or max depth
            current_depth = 0.0
            while True:
                if current_depth > max_depth_m:
                    print(f"[warn] depth cap reached at idx {pt['idx']}")
                    break

                # Next depth
                current_depth += step_m
                press = set_depth(appr, current_depth)
                movel(robot, press, a=cfg["PRESS_A"], v=cfg["PRESS_V"])

                # Clear sensor queues to avoid lag from old samples
                while not q_force.empty():
                    try: q_force.get_nowait()
                    except queue.Empty: break
                while not q_eit.empty():
                    try: q_eit.get_nowait()
                    except queue.Empty: break

                # Pause fully at this depth, then take a single sample
                time.sleep(cfg["SETTLE_S"])  # e.g., 3 seconds
                f_now = sample_force_now(q_force)
                e_now = sample_eit_now(q_eit)

                # Safety cutoff on force
                if f_now is not None and f_now > cfg["MAX_FORCE_N"]:
                    print(f"[safety] MAX_FORCE_N exceeded at idx {pt['idx']} ({f_now:.2f} N)")

                # Record one row for this depth step
                tcp = press
                rec = {
                    "t": datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                    "tcp_x": tcp[0], "tcp_y": tcp[1], "tcp_z": tcp[2],
                    "tcp_rx": tcp[3], "tcp_ry": tcp[4], "tcp_rz": tcp[5],
                    "contact_idx": pt["idx"],
                    "contact_u_m": pt["u_m"], "contact_v_m": pt["v_m"],
                    "contact_r_norm": pt["r"], "contact_theta": pt["theta"],
                    "depth_m": current_depth,
                    "force_n": f_now if f_now is not None else None,
                    "t_eit": e_now["t"] if e_now is not None and "t" in e_now else None,
                }
                # Fill EIT channels
                if eit_channel_count > 0:
                    # Pre-fill with None so columns exist even if no e_now
                    for i in range(eit_channel_count):
                        rec[f"eit_{i}"] = None
                if e_now is not None and "readings" in e_now:
                    for i, v in enumerate(e_now["readings"]):
                        rec[f"eit_{i}"] = v

                rows.append(rec)
                write_row_csv(rec)            # LIVE CSV append (includes EIT columns)
                maybe_write_parquet_chunk(rec)  # optional rolling parquet chunks

                # Stop when target force reached
                if f_now is not None and f_now >= cfg["TARGET_FORCE_N"]:
                    break

            # 3) Release straight up to approach plane before any lateral move
            movel(robot, appr, a=cfg["MOVE_A"], v=cfg["MOVE_V"])
            time.sleep(0.1)

        # End of plan
    except KeyboardInterrupt:
        print("[main] KeyboardInterrupt — homing to center plane and exiting...")
    finally:
        try:
            movel(robot, CONFIG["SENSOR_CENTER_POSE"], a=cfg["MOVE_A"], v=cfg["MOVE_V"])
            robot.close()
        except Exception:
            pass
        # Stop serial threads
        force_reader.stop(); eit_reader.stop()
        force_reader.join(timeout=1.0); eit_reader.join(timeout=1.0)

        # Save (finalization)
        if rows:
            # You already have a live CSV; we still write a final full Parquet snapshot.
            df = pd.DataFrame(rows)
            pq_path = str(run_base.with_suffix('.parquet'))
            df.to_parquet(pq_path, index=False)
            print(f"Saved Parquet → {pq_path}")

            # Flush any remaining parquet chunk buffer (if chunking enabled)
            if cfg.get("PARQUET_EVERY_N", 0) > 0 and len(parquet_buf) > 0:
                chunk_dir = Path(cfg["OUT_DIR"]) / f"{cfg['SESSION_TAG']}_{ts}_chunks"
                chunk_dir.mkdir(parents=True, exist_ok=True)
                chunk_path = chunk_dir / f"part_{parquet_chunk_id:05d}.parquet"
                pd.DataFrame(parquet_buf).to_parquet(chunk_path, index=False)
                print(f"Saved last chunk → {chunk_path}")

if __name__ == "__main__":
    main()
