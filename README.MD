# Euler Matrix Activation (EMA) 🌀

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vaggelis-mant/euler-matrix-activation/blob/main/example_usage.ipynb)
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](paper.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🛠️ Instructions & Quick Start##@
### 1. Run via Google Colab (Recommended)
The fastest way to verify the performance advantage of EMA is through Google Colab. 
* Click the **"Open In Colab"** badge at the top of this page.
* No local installation is required.

### 2. Local Installation
To use EMA in your local environment, ensure you have PyTorch installed:
```bash
pip install torch torchvision


3. Basic Usage
To integrate EMA into your own PyTorch models:

4. Reproduce Benchmark Results
To run the 30-layer deep comparison against GELU:

🚀 Key Performance Results
In our benchmarks using a 30-layer MLP (without residual connections) on MNIST:

Training Speed: EMA reached a loss of 0.4495 in the very first epoch, outperforming GELU's 0.6528.

Convergence Efficiency: Demonstrated a 31.1% improvement in initial convergence.

Deep Scaling: Maintains high signal stability even at extreme depths where standard activations struggle with erratic gradient flow.



🧬 Scientific Background
Euler Matrix Activation (EMA) is a novel, adaptive activation function based on complex-valued rotation matrices.


The Mathematical CoreInspired by Euler's formula $e^{i\theta} = \cos(\theta) + i\sin(\theta)$, EMA maps complex rotations to real $2 \times 2$ matrices for hardware-efficient execution:

$$f(x) = \alpha \cdot (x \cos(\theta) - \tanh(x) \sin(\theta))$$

By treating the activation as a learnable phase-shift, the network can dynamically adjust the "rotational state" of each neuron to maximize information flow and prevent vanishing gradients.




📜 Citation & Credits
Author: Evangelos Mantousis

Project: Adaptive Euler Matrix Activation (2026)

If you use this activation function in your research or commercial projects, please cite this repository.


⚖️ License
This project is licensed under the MIT License. See the  file for more details.
