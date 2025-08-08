import serial
import matplotlib.pyplot as plt
import time

SERIAL_PORT = "/dev/ttyUSB0"
BAUD_RATE = 115200

device = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)

device.write(b"d")
device.write(b"y")  # Trigger data stream if needed



# Plot setup
plt.ion()
fig, ax = plt.subplots()
line_plot, = ax.plot([], [], color='b', linewidth=1.5)
ax.set_ylim([0, 105500])
ax.set_title("EIT Real-time Plot")
ax.set_xlabel("Electrode Pair Index")
ax.set_ylabel("Magnitude")

buffer = ""
try:
    while True:
        if device.in_waiting:
            chunk = device.read(device.in_waiting).decode('utf-8', errors='ignore')
            buffer += chunk

            # Process all complete frames found in the buffer
            while "magnitudes:" in buffer:
                start_idx = buffer.find("magnitudes:")
                # Try to find end of frame (e.g., next 'magnitudes:' or a logical end based on commas count)
                next_start_idx = buffer.find("magnitudes:", start_idx + 1)

                if next_start_idx != -1:
                    frame = buffer[start_idx:next_start_idx].strip()
                    buffer = buffer[next_start_idx:]
                else:
                    # If no next frame detected yet, wait for more data
                    break

                # Process this frame
                raw_data = frame[len("magnitudes:"):].strip()
                try:
                    data = [float(x) for x in raw_data.split(',') if x.strip() != '']
                    if data:
                        print(f"Received frame with {len(data)} values")
                        line_plot.set_data(range(len(data)), data)
                        ax.set_xlim([0, len(data)])
                        plt.draw()
                        plt.pause(0.01)
                except ValueError:
                    print("Failed to parse frame")
                    continue

except KeyboardInterrupt:
    print("\nStopping...")

finally:
    device.close()
    plt.ioff()
    plt.show()
