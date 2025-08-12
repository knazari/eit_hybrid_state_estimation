import numpy as np
import matplotlib.pyplot as plt
import pyeit.eit.bp as bp
import pyeit.eit.protocol as protocol
import pyeit.mesh as mesh
from pyeit.mesh.shape import rectangle
from pyeit.eit.fem import EITForward
import serial
import time

# Your existing functions for reading sensor data
# def read_latest_line():
#     global EIT
#     buffer = b''
#     latest_line = None
#     while True:
#         if EIT.in_waiting > 0:
#             data = EIT.read(EIT.in_waiting)
#             buffer += data
#             lines = buffer.split(b'\n')
#             if len(lines) >= 1:
#                 if lines[-1] == b'':
#                     latest_line = lines[-2]
#                     buffer = b''
#             if latest_line is not None:
#                 return latest_line.decode()

_buf = b''  # persistent serial buffer

def read_latest_line():
    """
    Return the most recent complete CSV line from the serial port.
    Strips 'magnitudes:' prefix and whitespace. Safe with trailing comma.
    """
    global EIT, _buf

    while True:
        # Read whatever is available (at least 1 byte to avoid 0-sized reads)
        chunk = EIT.read(max(1, EIT.in_waiting or 1))
        if not chunk:
            # serial timeout hit; keep looping
            continue

        _buf += chunk

        # Process all complete lines in the buffer
        while b'\n' in _buf:
            line, _buf = _buf.split(b'\n', 1)  # take one complete line
            s = line.decode('utf-8', errors='ignore').strip()

            # Optional prefix from your stream
            if s.startswith("magnitudes:"):
                s = s[len("magnitudes:"):].strip()

            # Some firmware prints a trailing comma; safe to return anyway
            if s:
                return s

def split_eit_data(csv_line: str) -> np.ndarray:
    parts = [p.strip() for p in csv_line.split(',')]
    vals = []
    for p in parts:
        if not p:
            continue
        try:
            v = float(p)
            if v != 0.0:     # drop exact zeros from overlap pairs
                vals.append(v)
        except ValueError:
            pass
    return np.array(vals, dtype=np.float32)
            
# def split_eit_data(eit_data_str):
#     # The number of digits in each number (1 for the whole number part, 1 for the decimal point, and 4 for the decimal part)
#     digit_count = 6
#     # Number of numbers expected
#     num_numbers = len(eit_data_str) // digit_count
#     # Split the string into 6-character chunks and convert them to floats
#     voltage_data = [float(eit_data_str[i*digit_count:(i+1)*digit_count]) for i in range(num_numbers)]
#     voltage_data = np.array([float(num) for num in voltage_data if float(num) != 0])
#     return voltage_data

def process_data(data):
    voltage_data = [num.strip() for num in data.split(',') if num.strip()]
    # voltage_data = np.array([float(num) for num in voltage_data])
    voltage_data = np.array([float(num) for num in voltage_data if float(num) != 0])
    return voltage_data

def refresh():
    global v0
    v0 = read_latest_line()

# Setup serial port for EIT sensor
serial_port = '/dev/ttyACM0'
baud_rate = 115200
EIT = serial.Serial(serial_port, baud_rate, timeout=10)
print(f"Serial port {serial_port} opened successfully111.")

# EIT setup
n_el = 16  # number of electrodes
mesh_obj = mesh.create(n_el, h0=0.1)
protocol_obj = protocol.create(n_el, dist_exc=1, step_meas=1, parser_meas="std")

# Initialize the forward solver (for reference data)
fwd = EITForward(mesh_obj, protocol_obj)
v0 = read_latest_line()
v0 = read_latest_line()
v0 = read_latest_line()
v0 = read_latest_line()

v0 = read_latest_line()
# v0 = process_data(v0)
v0 = split_eit_data(v0)
v0 = v0
print(f"Reference data shape: {v0.shape}")
# v0 = v0.reshape(protocol_obj.n_exc, -1)  # reference data

# Initialize BP solver
eit_bp = bp.BP(mesh_obj, protocol_obj)
eit_bp.setup(weight="none")

# Initialize plot
fig, ax = plt.subplots(figsize=(6, 4))
plt.ion()

count = 0

# Extract node, element for plotting
pts = mesh_obj.node
tri = mesh_obj.element

def update_eit_image(voltage_data, message=None):
    # Reshape voltage data to match the protocol
    # v1 = voltage_data.reshape(protocol_obj.n_exc, -1)
    v1 = voltage_data
    
    # Solve the inverse problem
    # ds = 192.0 * eit_bp.solve(v1, v0, normalize=True, log_scale=False)

    subtraction = np.subtract(v0, v1)
    absolute_values = np.abs(subtraction)
    sum_absolute_values = np.sum(absolute_values)

    if sum_absolute_values > 0.3:
        
        ds = 192.0 * eit_bp.solve(v1, v0, normalize=True, log_scale=False)
    else:
        ds = 192.0 * eit_bp.solve(v0, v0, normalize=True)
    

    
    # Update the plot
    ax.clear()
    im = ax.tripcolor(pts[:, 0], pts[:, 1], tri, ds, shading="flat", cmap='viridis')
    ax.set_aspect('equal')
    ax.set_title('EIT Reconstruction (Multi-layered)')
    # plt.colorbar(im)

    if message:
        ax.text(0.5, 0.95, message, transform=ax.transAxes, ha='center', va='top', fontsize=24, color='red',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        plt.draw()
        plt.pause(5)
        
    
    plt.draw()
    plt.pause(0.001)
    # print(sum_absolute_values)

# Main loop
try:
    while True:
        # Read data from the sensor
        data = read_latest_line()
        # voltage_data = process_data(data)
        voltage_data = split_eit_data(data)

        # print(f"Received voltage data shape: {voltage_data.shape}")

        # Update EIT image
        # update_eit_image(voltage_data)
        # v0 = read_latest_line()
        # v0 = split_eit_data(v0)
        count = count + 1
        if count == 500:
            recalibration_message = "Re-calibrating\nPlease don't touch"
            update_eit_image(voltage_data, message=recalibration_message)
            
            v0 = read_latest_line()
            v0 = split_eit_data(v0)
            count = 0
            print(v0)
            continue
        else:
            update_eit_image(voltage_data)

        # time.sleep(0.1)  # Adjust this delay as needed

except KeyboardInterrupt:
    print("Stopping the program...")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    EIT.close()
    print("Serial port closed.")