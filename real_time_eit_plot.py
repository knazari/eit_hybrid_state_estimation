# import serial
# import matplotlib.pyplot as plt
# import time
# import numpy as np

# # Update this to match your Arduino port
# SERIAL_PORT = "/dev/ttyACM0"  # Use `ls /dev/ttyACM*` to check
# BAUD_RATE = 115200

# # Connect to Arduino
# device = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=5)

# # Wait briefly for connection to establish
# time.sleep(2)

# # Send trigger character
# device.write(b"y")

# # Read and discard the first line (initial baseline)
# raw = device.readline().decode('utf-8').strip()

# # Check and remove 'magnitudes:' prefix
# if raw.startswith("magnitudes:"):
#     raw = raw[len("magnitudes:"):].strip()

# try:
#     baseline = [float(x) for x in raw.strip().split(',') if x.strip() != '']
# except:
#     baseline = []

# # print("Baseline:", baseline)

# # Plot setup
# plt.ion()
# fig, ax = plt.subplots()
# line_plot, = ax.plot([], [], color='b', linewidth=1.5)
# ax.set_ylim([0.0, 0.8])
# ax.set_title("EIT Real-time Plot")
# ax.set_xlabel("Electrode Pair Index")
# ax.set_ylabel("Magnitude")

# # Real-time loop
# try:
#     i = 0
#     while True:
#         raw = device.readline().decode('utf-8').strip()
#         # Check and remove 'magnitudes:' prefix
#         # print(f"[{i}] Raw line: '{raw}'")  # Debug print

#         try:
#             data = [float(x) for x in raw.strip().split(',') if x.strip() != '']
#             if data:
#                 line_plot.set_data(range(len(data)), data)
#                 ax.set_xlim([0, len(data)])
#                 plt.draw()
#                 plt.pause(0.01)
#         except ValueError:
#             continue
        
#         i += 1

# except KeyboardInterrupt:
#     print("\nInterrupted by user. Closing serial connection...")

# finally:
#     device.close()
#     plt.ioff()
#     plt.show()







import serial
import matplotlib.pyplot as plt
import time
import numpy as np

# ======================
# CONFIG
# ======================
SERIAL_PORT = "/dev/ttyACM0"   # e.g. /dev/ttyACM0
BAUD_RATE = 115200
Y_LIMITS = [-0.2, 0.4]         # adjust depending on signal range

# ======================
# CONNECT
# ======================
device = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=5)
time.sleep(2)  # allow Arduino reset

# Request baseline frame
device.write(b"y")
time.sleep(0.2)
raw = device.readline().decode("utf-8", errors="ignore").strip()
if raw.startswith("magnitudes:"):
    raw = raw[len("magnitudes:"):].strip()

# Parse baseline
try:
    baseline = np.array([float(x) for x in raw.split(",") if x.strip() != ""], dtype=np.float32)
    # Drop zeros (e.g. unused channels)
    baseline = baseline[baseline != 0]
except Exception as e:
    print(f"⚠️ Failed to parse baseline: {e}")
    baseline = np.zeros(256, dtype=np.float32)

print(f"✅ Baseline captured with {len(baseline)} channels")

# ======================
# PLOT SETUP
# ======================
plt.ion()
fig, ax = plt.subplots(figsize=(8, 4))
line_plot, = ax.plot([], [], color="tab:blue", lw=1.5)
ax.set_ylim(Y_LIMITS)
ax.set_xlim(0, len(baseline))
ax.set_title("EIT Real-Time ΔV (Signal - Baseline)")
ax.set_xlabel("Electrode Pair Index")
ax.set_ylabel("Δ Magnitude (V - V₀)")

# ======================
# LOOP
# ======================
try:
    while True:
        raw = device.readline().decode("utf-8", errors="ignore").strip()
        if not raw:
            continue
        if raw.startswith("magnitudes:"):
            raw = raw[len("magnitudes:"):].strip()

        try:
            data = np.array([float(x) for x in raw.split(",") if x.strip() != ""], dtype=np.float32)
            data = data[data != 0]
            if len(data) == len(baseline):
                delta = data - baseline
                line_plot.set_data(np.arange(len(delta)), delta)
                ax.set_xlim(0, len(delta))
                plt.draw()
                plt.pause(0.01)
        except Exception:
            continue

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user. Closing serial connection...")

finally:
    device.close()
    plt.ioff()
    plt.show()
