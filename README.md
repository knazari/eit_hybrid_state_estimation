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
* NumPy
* SciPy
* PyTorch
* Matplotlib

All dependencies are listed in `requirements.txt`.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For questions, collaborations, or data requests, contact **Kiyanoush Nazari** at \[[k.sasikolomi@ucl.ac.uk](mailto:k.sasikolomi@ucl.ac.uk)].

