# ⚛️ Quantum Neural Hybrid Classifier (QML)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-Quantum_Sim-purple.svg)](https://pennylane.ai/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive_App-FF4B4B.svg)](https://streamlit.io/)

A high-performance **Hybrid Quantum-Classical Machine Learning Architecture** merging quantum state representations with classical deep learning optimization pipelines. This model encodes multidimensional feature inputs into variational quantum circuits (PQCs) and computes gradients using PyTorch autograd integration.

---

## 🖼️ Interactive Dashboard & System Previews

Here is the live interface demonstrating quantum circuit state evaluation across different feature configurations:

### 1. Quantum State Vector & Execution Engine (Class 1 Output)
<p align="center">
  <img src="assets/qml-dashboard-class1" alt="QML Dashboard Class 1 Output" width="90%"/>
</p>

### 2. Parameter Tuning & Real-time Inference (Class 0 Output)
<p align="center">
  <img src="assets/qml-dashboard-class0" alt="QML Dashboard Class 0 Output" width="90%"/>
</p>

### 3. Interactive Probability Superposition Spectrum
<p align="center">
  <img src="assets/interactive-spectrum" alt="Interactive Probability Spectrum Plotly" width="90%"/>
</p>

---

## 🚀 Key Features

* **Quantum Embeddings**: Dynamic multi-qubit encoding using `qml.AngleEmbedding` to map real-valued classical feature vectors into Hilbert Space.
* **Variational Quantum Circuit (VQC)**: Multi-layer parametrized entangling gates (`qml.StronglyEntanglingLayers`) for learning state rotations[cite: 1].
* **Backprop Gradient Pipeline**: Stable batch training using PennyLane's `diff_method="backprop"` interface on `default.qubit`[cite: 1].
* **Interactive Sci-Fi Dashboard**: Built with Streamlit and Plotly for real-time visualization of quantum state probability distributions.

---

## 🛠️ Architecture & Tech Stack

```text
qml-hybrid-classifier/
│
├── assets/                     # UI Screenshots & visual assets
│   ├── qml-dashboard-class1
│   ├── qml-dashboard-class0
│   └── interactive-spectrum
│
├── model.py                    # Quantum-Classical Hybrid Model Definition (PennyLane + PyTorch)[cite: 1]
├── train.py                    # Synthetic dataset generation & gradient backpropagation loop
├── app.py                      # Sci-Fi Interactive Streamlit web interface
├── requirements.txt            # Project dependency declaration[cite: 1]
└── .gitignore                  # Exclusion rules for venv, cache, and build artifacts