
---

## 📁 Folder 2: `model/README.md`

**Content (copy-paste):**

```markdown
# Model Files

This folder contains the trained ECG-TransCovNet model files.

## Files

| File | Description | Size |
|------|-------------|------|
| `ecg_transcovnet.keras` | Full trained model | 1.38 MB |
| `ecg_transcovnet_int8.tflite` | INT8 quantized model | 0.38 MB |
| `architecture.png` | Model architecture diagram | — |

## Model Details

| Parameter | Value |
|-----------|-------|
| Input | 259 samples (ECG beat) + 4 RR-interval features |
| Output | 4 classes (N, S, V, Q) |
| Total Parameters | ~408,000 |
| Model Size (Keras) | 1.38 MB |
| Model Size (INT8) | 0.38 MB |

## Architecture

- **CNN Branch:** 3 conv layers (64→128→256 filters, kernels 7→5→3)
- **Transformer Branch:** 2 encoder layers, 4 attention heads, 128-dim
- **Attention Fusion:** Dynamic weighting (α_cnn + α_trans = 1)
- **RR Branch:** 4 RR-interval features → Dense(32) → Dense(16)
- **Classification Head:** 4 classes (N, S, V, Q)

## Usage

```python
import tensorflow as tf

# Load full model
model = tf.keras.models.load_model('ecg_transcovnet.keras')

# Load INT8 quantized model

Performance
Class	Type	F1-Score
N	Normal	0.957
S	Supraventricular	0.152
V	Ventricular	0.853
Q	Paced/Unknown	0.623
interpreter = tf.lite.Interpreter(model_path='ecg_transcovnet_int8.tflite')
interpreter.allocate_tensors()
