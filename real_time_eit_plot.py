import serial
import matplotlib.pyplot as plt
import time

# Update this to match your Arduino port
SERIAL_PORT = "/dev/ttyACM0"  # Use `ls /dev/ttyACM*` to check
BAUD_RATE = 115200
NUM_READS = 300

# Connect to Arduino
device = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=25)

# Wait briefly for connection to establish
time.sleep(2)

# Send trigger character
device.write(b"y")

# Read and discard the first line (initial baseline)
line = device.readline().decode('utf-8').strip()
try:
    baseline = [float(x) for x in raw.strip().split(',') if x.strip() != '']
except:
    baseline = []

print("Baseline:", baseline)

# Plot setup
plt.ion()
fig, ax = plt.subplots()
line_plot, = ax.plot([], [], color='b', linewidth=1.5)
ax.set_ylim([0, 0.8])
ax.set_title("EIT Real-time Plot")
ax.set_xlabel("Electrode Pair Index")
ax.set_ylabel("Magnitude")

# Read and plot in real time
for i in range(NUM_READS):
    raw = device.readline().decode('utf-8').strip()
    # print(f"[{i}] Raw line: '{raw}'")  # <-- Add this line

    try:
        data = [float(x) for x in raw.strip().split(',') if x.strip() != '']
        # print(data)
        if data:
            line_plot.set_data(range(len(data)), data)
            ax.set_xlim([0, len(data)])
            plt.draw()
            plt.pause(0.01)
    except ValueError:
        continue


# Close serial
device.close()
plt.ioff()
plt.show()
