import serial
import matplotlib.pyplot as plt
import time
import numpy as np

# Update this to match your Arduino port
SERIAL_PORT = "/dev/ttyACM0"  # Use `ls /dev/ttyACM*` to check
BAUD_RATE = 115200

# Connect to Arduino
device = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=5)

# Wait briefly for connection to establish
time.sleep(2)

# Send trigger character
device.write(b"y")

# Read and discard the first line (initial baseline)
raw = device.readline().decode('utf-8').strip()

# Check and remove 'magnitudes:' prefix
if raw.startswith("magnitudes:"):
    raw = raw[len("magnitudes:"):].strip()

try:
    baseline = [float(x) for x in raw.strip().split(',') if x.strip() != '']
except:
    baseline = []

# print("Baseline:", baseline)

# Plot setup
plt.ion()
fig, ax = plt.subplots()
line_plot, = ax.plot([], [], color='b', linewidth=1.5)
ax.set_ylim([0.0, 1.0])
ax.set_title("EIT Real-time Plot")
ax.set_xlabel("Electrode Pair Index")
ax.set_ylabel("Magnitude")

# Real-time loop
try:
    i = 0
    while True:
        raw = device.readline().decode('utf-8').strip()
        # Check and remove 'magnitudes:' prefix
        # print(f"[{i}] Raw line: '{raw}'")  # Debug print

        try:
            data = [float(x) for x in raw.strip().split(',') if x.strip() != '']
            # new_data = [a_i - b_i for a_i, b_i in zip(data, baseline)]
            # print(data)
            # new_data = [x for x in data if x != 0]
            if data:
                line_plot.set_data(range(len(data)), data)
                ax.set_xlim([0, len(data)])
                plt.draw()
                plt.pause(0.01)
        except ValueError:
            continue
        
        i += 1

except KeyboardInterrupt:
    print("\nInterrupted by user. Closing serial connection...")

finally:
    device.close()
    plt.ioff()
    plt.show()
