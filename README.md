# ECG-TransCovNet

A Hybrid CNN-Transformer Architecture for Multi-Class Arrhythmia Detection in ECG Signals

[![Thesis](https://img.shields.io/badge/Thesis-PDF-blue)](thesis/ECG-TransCovNet_Thesis.pdf)
[![Python](https://img.shields.io/badge/Python-3.10-green)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

ECG-TransCovNet is a lightweight hybrid CNN-Transformer architecture for multi-class arrhythmia detection in ECG signals. The model is designed for resource-constrained healthcare settings like rural Pakistan, where cardiologists are scarce and computing resources are limited.

The research follows a two-phase methodology:

1. Synthetic data for architectural validation
2. Real MIT-BIH data for clinical validation with strict subject-disjoint splits

---

## Key Results

| Metric | Value |
|--------|-------|
| Accuracy (Real MIT-BIH) | 91.39% |
| Macro-F1 | 0.646 |
| Weighted-F1 | 0.910 |
| Parameters | ~408,000 |
| Model Size (Keras) | 1.38 MB |
| Model Size (INT8) | 0.38 MB |
| CPU Inference | 5.8 ms |
| INT8 Inference | 4.21 ms |

---

## Repository Structure

ECG-TransCovNet/
├── thesis/              # Thesis PDF
├── model/               # Trained model files
├── code/                # Source code
├── results/             # Result plots
├── notebooks/           # Jupyter notebooks
├── LICENSE
└── README.md

---

## Model Architecture

- CNN Branch: 3 conv layers (64→128→256 filters, kernels 7→5→3)
- Transformer Branch: 2 encoder layers, 4 attention heads, 128-dim
- Attention Fusion: Dynamic weighting (α_cnn + α_trans = 1)
- RR Branch: 4 RR-interval features → Dense(32) → Dense(16)
- Classification Head: 4 classes (N, S, V, Q)

---

## Per-Class Performance (Real MIT-BIH)

| Class | Type | Precision | Recall | F1-Score | AUROC |
|-------|------|-----------|--------|----------|-------|
| N | Normal | 0.969 | 0.945 | 0.957 | 0.958 |
| S | Supraventricular | 0.355 | 0.097 | 0.152 | 0.861 |
| V | Ventricular | 0.781 | 0.939 | 0.853 | 0.986 |
| Q | Paced/Unknown | 0.471 | 0.921 | 0.623 | 0.990 |

---

## Dataset

MIT-BIH Arrhythmia Database

- 48 records, 47 subjects
- 109,494 beats
- 360 Hz sampling rate
- MLII single-lead
- 4-class AAMI (N, S, V, Q)

---

## Technologies

- Python 3.10
- TensorFlow 2.15 / Keras
- Scikit-learn
- NumPy / SciPy
- Matplotlib
- Google Colab

---

## Quick Start

### Clone Repository

git clone https://github.com/Shoaibdraban/ECG-TransCovNet.git
cd ECG-TransCovNet

### Install Dependencies

pip install -r code/requirements.txt

### Load Model

import tensorflow as tf

model = tf.keras.models.load_model('model/ecg_transcovnet.keras')

interpreter = tf.lite.Interpreter(model_path='model/ecg_transcovnet_int8.tflite')
interpreter.allocate_tensors()

---

## Thesis

Title: ECG-TransCovNet: A Hybrid CNN-Transformer Architecture for Multi-Class Arrhythmia Detection in ECG Signals

Author: Muhammad Shoaib
Supervisor: Dr. Khalid Mehmood
University: Gomal University, Dera Ismail Khan
Year: 2024–2026

Download Thesis PDF: [thesis/ECG-TransCovNet_Thesis.pdf](thesis/ECG-TransCovNet_Thesis.pdf)

---

## Citation

If you use this work in your research, please cite:

@mastersthesis{shoaib2026ecg,
  title={ECG-TransCovNet: A Hybrid CNN-Transformer Architecture for Multi-Class Arrhythmia Detection in ECG Signals},
  author={Shoaib, Muhammad},
  year={2026},
  school={Gomal University, Dera Ismail Khan}
}

---

## Contact

Muhammad Shoaib
Email: shoaibdraban@gmail.com
WhatsApp: +92 346 7851061
GitHub: https://github.com/Shoaibdraban
Portfolio: https://shoaibdraban-portfolio.netlify.app

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
