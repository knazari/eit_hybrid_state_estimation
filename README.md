# Hybrid State Estimation of EIT Tactile Sensor

This repository contains the code and implementation for a hybrid state estimation framework applied to Electrical Impedance Tomography (EIT)-based tactile sensors for robotic manipulation tasks.

## Project Overview
Electrical Impedance Tomography (EIT) tactile sensors provide rich spatial deformation information through voltage measurements. This project focuses on developing a hybrid estimation approach that combines data-driven learning models with model-based filtering techniques to infer contact force distributions and object interaction states in real-time.

## Features
- EIT forward model simulation using [PyEIT](https://github.com/pyEIT/pyEIT).
- Data-driven state estimation using neural networks.
- Kalman Filter integration for hybrid estimation.
- Visualization tools for voltage patterns and estimated contact forces.
- Modular code structure for ease of testing and experimentation.

## Repository Structure
```

├── src/                  # Source code (models, estimators, simulation scripts)
├── data/                 # Calibration and testing data (excluded from Git tracking)
├── notebooks/            # Jupyter notebooks for experimentation and visualization
├── utils/                # Helper scripts (visualization tools, utilities)
├── README.md             # Project documentation
└── requirements.txt      # Python dependencies

````

> **Note**: The `data/` folder is not included in this repository due to size constraints. Contact the author for access to the dataset.

## Installation
Clone the repository and install required Python packages:

```bash
git clone https://github.com/your-username/Hybrid-State-Estimation-EIT-Tactile.git
cd Hybrid-State-Estimation-EIT-Tactile
pip install -r requirements.txt
````

## 🧠 Teensy Board Installation on Linux (Tested on Ubuntu)

This guide walks you through installing the **Arduino IDE (v1.8.19)** and setting it up to work with the **Teensy board** on Linux.

---

### 📥 Step 1: Download Arduino IDE 1.8.19

Download and unzip Arduino IDE version **1.8.19** from the official Arduino website:

➡️ [https://www.arduino.cc/en/software/](https://www.arduino.cc/en/software/)

Choose the **Linux 64-bit** version (`.tar.xz`).

---

### ⚙️ Step 2: Install udev Rules for Teensy

Teensy requires a udev rule so it can be accessed without root permissions.

1. Go to: [https://www.pjrc.com/teensy/td_download.html](https://www.pjrc.com/teensy/td_download.html)
2. Scroll down and find the **udev rule** text.
3. Create a new file:
nano 00-teensy.rules
4. Paste the rule contents into the file and save it.

5. Move the rule into the correct system directory:

         sudo cp 00-teensy.rules /etc/udev/rules.d/

### 🔧 Step 3: Install Teensyduino

Download the Teensyduino Linux Installer (X86 64-bit) from the same PJRC download page.
Give it permission and run:

      chmod 755 TeensyduinoInstall.linux64 ./TeensyduinoInstall.linux64

When prompted, select the previously extracted Arduino 1.8.19 folder.

✅ Final Step: Verify Installation

Launch the Arduino IDE (arduino inside the unzipped folder). Connect your Teensy board via USB.
 You should now see "Teensy" listed under Tools > Board.

📝 Notes

This setup only works with Arduino 1.8.19 (not the newer 2.x versions).

If the Teensy board doesn't show up, try restarting your system or unplugging/replugging the board.
You may need to install additional packages like libusb if prompted.

## Usage

1. **Run EIT simulation and generate voltage patterns:**

   ```bash
   python src/simulate_eit.py
   ```

2. **Train the hybrid state estimator:**

   ```bash
   python src/train_hybrid_estimator.py
   ```

3. **Visualize contact estimation results:**

   ```bash
   python utils/visualize_estimation.py
   ```

## Dependencies

* Python 3.9+
* PyEIT
* Arduino 1.8.19
* NumPy
* SciPy
* PyTorch
* Matplotlib

All dependencies are listed in `requirements.txt`.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions, collaborations, or data requests, contact **Kiyanoush Nazari** at \[[k.sasikolomi@ucl.ac.uk](mailto:k.sasikolomi@ucl.ac.uk)].

