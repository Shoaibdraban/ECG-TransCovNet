# ECG-TransCovNet

A Hybrid CNN-Transformer Architecture for Multi-Class Arrhythmia Detection in ECG Signals

[![Thesis](https://img.shields.io/badge/Thesis-PDF-blue)](https://github.com/Shoaibdraban/ECG-TransCovNet/blob/main/ECG-TransCovNet_Thesis.pdf)
[![Python](https://img.shields.io/badge/Python-3.10-green)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-3.15-red)](https://keras.io)
[![TFLite](https://img.shields.io/badge/TFLite-170%20KB%20INT8-brightgreen)](https://ai.google.dev/edge/lite)
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
| Model Size (TFLite float32) | 430 KB |
| Model Size (TFLite INT8) | **170 KB** |
| CPU Inference | 5.8 ms |
| Raspberry Pi 4 | 18.7 ms/beat |

---

## Repository Structure

```
ECG-TransCovNet/
├── ECG-TransCovNet_Thesis.pdf      # Thesis (2.5 MB)
├── thesis/                          # Thesis-related files
├── model/                           # Trained models + TFLite
├── code/                            # Source code
├── results/                         # Result plots
├── notebooks/                       # Jupyter notebooks
├── LICENSE
└── README.md
```

---

## Model Architecture

- **CNN Branch:** 3 conv layers (64→128→256 filters, kernels 7→5→3)
- **Transformer Branch:** 2 encoder layers, 4 attention heads, 128-dim feed-forward
- **Attention Fusion:** Dynamic weighting (α_cnn + α_trans = 1)
- **RR Branch:** 4 RR-interval features → Dense(32) → Dense(16)
- **Classification Head:** 4 classes (N, S, V, Q)

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

```bash
git clone https://github.com/Shoaibdraban/ECG-TransCovNet.git
cd ECG-TransCovNet
```

### Install Dependencies

```bash
pip install -r code/requirements.txt
```

### Load Keras Model

```python
import numpy as np
import keras
from keras import layers, ops

@keras.saving.register_keras_serializable(package="ECGTransCovNet")
class AddPosEncoding(layers.Layer):
    def __init__(self, pos_enc_value, **kwargs):
        super().__init__(**kwargs)
        self.pos_enc_value = pos_enc_value
    def call(self, t):
        return t + ops.convert_to_tensor(self.pos_enc_value)
    def get_config(self):
        cfg = super().get_config()
        cfg["pos_enc_value"] = self.pos_enc_value.tolist()
        return cfg

model = keras.models.load_model(
    "model/ecg_rebuilt.keras",
    custom_objects={"AddPosEncoding": AddPosEncoding},
    safe_mode=False,
)

waveform    = np.random.randn(1, 259, 1).astype(np.float32)
rr_features = np.random.randn(1, 4).astype(np.float32)
preds = model.predict([waveform, rr_features], verbose=0)
```

### Load TFLite Model

```python
import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="model/ecg_transcovnet_dynamic.tflite")
interpreter.allocate_tensors()

inputs  = interpreter.get_input_details()
outputs = interpreter.get_output_details()

for d in inputs:
    if d["shape"][1] == 259:
        waveform_idx = d["index"]
    elif d["shape"][1] == 4:
        rr_idx = d["index"]

waveform = np.random.randn(1, 259, 1).astype(np.float32)
rr       = np.random.randn(1, 4).astype(np.float32)

interpreter.set_tensor(waveform_idx, waveform)
interpreter.set_tensor(rr_idx, rr)
interpreter.invoke()

preds = interpreter.get_tensor(outputs[0]["index"])  # shape (1, 4)
```

---

## Thesis

**Title:** ECG-TransCovNet: A Hybrid CNN-Transformer Architecture for Multi-Class Arrhythmia Detection in ECG Signals

**Author:** Muhammad Shoaib  
**Supervisor:** Dr. Khalid Mehmood  
**University:** Gomal University, Dera Ismail Khan  
**Year:** 2024–2026

📄 **[Download Thesis PDF](https://github.com/Shoaibdraban/ECG-TransCovNet/blob/main/ECG-TransCovNet_Thesis.pdf)**

---

## Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{shoaib2026ecg,
  title  = {ECG-TransCovNet: A Hybrid CNN-Transformer Architecture for Multi-Class Arrhythmia Detection in ECG Signals},
  author = {Shoaib, Muhammad},
  year   = {2026},
  school = {Gomal University, Dera Ismail Khan}
}
```

---

## Contact

**Muhammad Shoaib**  
📧 Email: shoaibdraban@gmail.com  
💬 WhatsApp: +92 346 7851061  
🐙 GitHub: [@Shoaibdraban](https://github.com/Shoaibdraban)  
🌐 Portfolio: [shoaibdraban-portfolio.netlify.app](https://shoaibdraban-portfolio.netlify.app)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
