# Model Files

This folder contains the trained ECG-TransCovNet model files.

---

## Files

| File | Description | Size |
|------|-------------|------|
| ecg_transcovnet.keras | Full trained model | 1.38 MB |
| ecg_transcovnet_int8.tflite | INT8 quantized model | 0.38 MB |
| architecture.png | Model architecture diagram | — |

---

## Model Details

| Parameter | Value |
|-----------|-------|
| Input | 259 samples (ECG beat) + 4 RR-interval features |
| Output | 4 classes (N, S, V, Q) |
| Total Parameters | ~408,000 |
| Model Size (Keras) | 1.38 MB |
| Model Size (INT8) | 0.38 MB |
| CPU Inference | 5.8 ms |
| INT8 Inference | 4.21 ms |

---

## Architecture

### CNN Branch (Local Feature Extraction)

- 3 conv layers (64 to 128 to 256 filters)
- Kernels: 7 to 5 to 3
- BatchNorm + ReLU + MaxPool after each layer
- Global Average Pooling at the end

### Transformer Branch (Global Context Modeling)

- 2 encoder layers
- 4 attention heads
- 128-dim embedding
- Learned positional encoding
- Feed-forward dim: 512

### Attention-Based Fusion

- Dynamic weighting: alpha_cnn + alpha_trans = 1
- Weighted sum of CNN (256-dim) + Transformer (128-dim) features

### RR Branch

- 4 RR-interval features
- Dense(32) then Dense(16)

### Classification Head

- Dense(64) + Dropout(0.4)
- Output: Dense(4) with Softmax

---

## Usage

### Load Full Model

import tensorflow as tf

model = tf.keras.models.load_model('ecg_transcovnet.keras')
model.summary()

### Load INT8 Quantized Model

import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter(model_path='ecg_transcovnet_int8.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_data = np.array([...], dtype=np.float32)
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])

---

## Performance

### Per-Class F1-Score (Real MIT-BIH)

| Class | Type | F1-Score |
|-------|------|----------|
| N | Normal | 0.957 |
| S | Supraventricular | 0.152 |
| V | Ventricular | 0.853 |
| Q | Paced/Unknown | 0.623 |

### Overall Performance

| Metric | Value |
|--------|-------|
| Accuracy | 91.39% |
| Macro-F1 | 0.646 |
| Weighted-F1 | 0.910 |

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Dataset | MIT-BIH Arrhythmia Database |
| Records | 48 records, 47 subjects |
| Total Beats | 109,494 |
| Sampling Rate | 360 Hz |
| Lead | MLII (single-lead) |
| Split | Subject-disjoint |
| Cross-Validation | 5-fold GroupKFold |
| Random Seed | 42 |

---

## Attention Weights by Class

| Class | alpha_cnn | alpha_trans | Dominant Branch |
|-------|-----------|-------------|-----------------|
| N | 0.52 | 0.48 | Balanced |
| S | 0.48 | 0.52 | Balanced |
| V | 0.71 | 0.29 | CNN |
| Q | 0.32 | 0.68 | Transformer |

---

## Citation

If you use this model in your research, please cite:

@mastersthesis{shoaib2026ecg,
  title={ECG-TransCovNet: A Hybrid CNN-Transformer Architecture for Multi-Class Arrhythmia Detection in ECG Signals},
  author={Shoaib, Muhammad},
  year={2026},
  school={Gomal University, Dera Ismail Khan}
}

---

## Related Resources

- Code Repository: https://github.com/Shoaibdraban/ECG-TransCovNet
- Thesis PDF: ../thesis/ECG-TransCovNet_Thesis.pdf
- Portfolio: https://shoaibdraban-portfolio.netlify.app

---

## Contact

Muhammad Shoaib
Email: shoaibdraban@gmail.com
GitHub: https://github.com/Shoaibdraban
Portfolio: https://shoaibdraban-portfolio.netlify.app

---

## License

This model is licensed under the MIT License — see the LICENSE file for details.
