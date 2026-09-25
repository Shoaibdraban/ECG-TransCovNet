# ECG-TransCovNet — Model Files

[![Thesis](https://github.com/Shoaibdraban/ECG-TransCovNet/blob/main/ECG-TransCovNet_Thesis.pdf)
[![Python](https://img.shields.io/badge/Python-3.10-green)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-3.15-red)](https://keras.io)
[![TFLite](https://img.shields.io/badge/TFLite-170%20KB%20INT8-brightgreen)](https://ai.google.dev/edge/lite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

This folder contains the trained ECG-TransCovNet model and its deployment files.

## Files

| File | Size | Description |
|------|------|-------------|
| `ecg_rebuilt.keras` | ~1.4 MB | Full Keras model (architecture + trained weights) |
| `baseline_a_final_locked.keras` | ~633 KB | Baseline CNN for comparison |
| `ecg_transcovnet_fp32.tflite` | ~430 KB | TFLite float32 — bit-exact with Keras (max diff < 1e-7) |
| `ecg_transcovnet_dynamic.tflite` | **~170 KB** | TFLite dynamic-range INT8 — for edge / mobile deployment |

## Verified

- **Keras ↔ float32 TFLite:** max output difference `1.19e-07` (float32 precision noise)
- **Keras ↔ dynamic INT8 TFLite:** max probability shift `~0.01`; argmax (predicted class) identical
- **Input shapes:** `waveform (259, 1)` + `rr_features (4,)`
- **Output shape:** `(4,)` softmax over classes `[N, S, V, Q]` (AAMI)
- **Dataset:** MIT-BIH Arrhythmia Database (48 records, 109,494 beats, subject-disjoint split)

## Usage — Keras

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
    "ecg_rebuilt.keras",
    custom_objects={"AddPosEncoding": AddPosEncoding},
    safe_mode=False,
)

waveform    = np.random.randn(1, 259, 1).astype(np.float32)
rr_features = np.random.randn(1, 4).astype(np.float32)

preds = model.predict([waveform, rr_features], verbose=0)
# preds.shape == (1, 4), sum ~= 1.0
```

## Usage — TFLite float32 (bit-exact)

```python
import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter("ecg_transcovnet_fp32.tflite")
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

preds = interpreter.get_tensor(outputs[0]["index"])  # (1, 4)
```

## Usage — TFLite dynamic INT8 (170 KB, edge deployment)

Use this when you need the smallest possible model for Raspberry Pi,
mobile devices, or microcontrollers.

```python
import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter("ecg_transcovnet_dynamic.tflite")
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

preds = interpreter.get_tensor(outputs[0]["index"])  # (1, 4)
```

**Trade-off vs float32 TFLite:**
- ✅ ~60% smaller (170 KB vs 430 KB)
- ✅ Argmax (predicted class) identical to Keras
- ⚠️ Max probability shift ~0.01 (int8 weight rounding)

For applications where bit-exact probabilities matter, use
`ecg_transcovnet_fp32.tflite` instead.

## Performance

| Metric | Value |
|--------|-------|
| Accuracy (locked test set) | **91.39%** |
| Macro F1 | 0.646 |
| Weighted F1 | 0.910 |
| Parameters | ~408,000 |
| Keras size | 1.38 MB |
| TFLite float32 size | 430 KB |
| TFLite INT8 size | **170 KB** |
| CPU inference | ~5.8 ms/beat |
| Raspberry Pi 4 | 18.7 ms/beat |

### Per-Class F1

| Class | F1 |
|-------|----|
| N (Normal) | 0.957 |
| S (Supraventricular) | 0.152 |
| V (Ventricular) | 0.853 |
| Q (Paced / Unknown) | 0.623 |

## Architecture

- **CNN Branch:** 3 × Conv1D (64 → 128 → 256 filters, kernels 7 → 5 → 3) + BatchNorm + MaxPool + Dropout
- **Transformer Branch:** 2 encoder blocks, 4 attention heads, 128-dim feed-forward
- **Attention Fusion:** dynamic weighting (α_cnn + α_trans = 1)
- **RR Branch:** 4 RR-interval features → Dense(32) → Dense(16)
- **Classification Head:** Dense(4) softmax

## Citation

If you use these model files in your research, please cite:

```bibtex
@mastersthesis{shoaib2026ecg,
  title  = {ECG-TransCovNet: A Hybrid CNN-Transformer Architecture for Multi-Class Arrhythmia Detection in ECG Signals},
  author = {Shoaib, Muhammad},
  year   = {2026},
  school = {Gomal University, Dera Ismail Khan}
}
```

## License

MIT — see [LICENSE](../LICENSE) at repository root.
