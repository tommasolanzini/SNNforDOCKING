# Bio-Inspired Spiking Neural Networks for Spacecraft Docking
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![snnTorch](https://img.shields.io/badge/snnTorch-enabled-brightgreen)](https://snntorch.readthedocs.io/)
A 4-layer Spiking Neural Network (SNN) performing continuous 6DoF pose estimation for autonomous spacecraft rendezvous using event-based vision (Dynamic Vision Sensors).

## ⚠️ Requirements
* **Hardware:** NVIDIA GPU with **CUDA** support.
* **OS:** Linux / Windows 10/11
* **Python:** 3.8+

## 🛠️ Dependencies & Installation
The project uses the following libraries:
* `torch`
* `snntorch`
* `numpy`
* `pandas`
* `matplotlib`

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tommasolanzini/SNNforDOCKING.git

2. **Download dataset:**
   https://zenodo.org/records/7370076


##  Usage & Reproducibility

  The repository workflow is highly modular. You can either train the networks entirely from scratch or bypass training by directly evaluating the pre-trained models provided in the `NNs_weights/` folder.
  
  ### 1. Training (ANN & SNN)
  You can train either architecture directly from scratch or apply the ANN-to-SNN transfer learning paradigm.
  * **`SNN.py` / ANN scripts:** Execute the main scripts to initiate the training loop. Ensure CUDA is enabled for tractable processing of the spatiotemporal tensors.
  
  ### 2. Testing & Visualization
  Bypass training and directly test the network's capabilities on the unseen test dataset using the provided weights.
  * **`testing_net`**: Loads pre-trained weights from the `NNs_weights/` directory and performs inference. It plots the predicted 6DoF trajectory against the ground truth to visualize how accurately the network tracks the real approach.
  
  ### 3. Statistical Evaluation
  * **`statistical_eval`**: Performs rigorous statistical analysis on the pre-trained test runs. It evaluates the 10 independent models to compute overall uncertainties, generating the RMSE and MSE error bar plots per Degree of Freedom.
  
  ### 4. Threshold & Hyperparameter Comparison
  * **`threshold`**: Compares the network's tracking performance across different LIF membrane thresholds. It utilizes specific weights to demonstrate how threshold tuning impacts voltage propagation and prevents signal attenuation.
  
  ### 5. Radiation Fault Injection
  * **`radiation_inj`**: Simulates deployment in a Low Earth Orbit (LEO) environment. It injects cosmic radiation faults (Single Event Upsets / bit-flips) into the hardware memory of both the ANN and SNN, directly comparing their structural reliability and catastrophic failure rates.
