# ECG-TransCovNet — Model Files

This folder contains the trained ECG-TransCovNet model and its deployment files.

## Files

| File | Size | Description |
|------|------|-------------|
| `ecg_rebuilt.keras` | ~1.4 MB | Full Keras model (architecture + trained weights) |
| `baseline_a_final_locked.keras` | ~633 KB | Baseline CNN for comparison |
| `ecg_transcovnet_fp32.tflite` | ~430 KB | TFLite deployment file for edge / mobile / Raspberry Pi |

## Verified

- **Keras ↔ TFLite max output difference:** `1.19e-07` (float32 numerical precision noise)
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

## Usage — TFLite (Python / Edge / Raspberry Pi)

```python
import numpy as np
import tensorflow as tf

interpreter = tf.lite.Interpreter("ecg_transcovnet_fp32.tflite")
interpreter.allocate_tensors()

inputs  = interpreter.get_input_details()
outputs = interpreter.get_output_details()

# Auto-detect which input is which
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

## Performance

| Metric | Value |
|--------|-------|
| Accuracy (locked test set) | **91.39%** |
| Macro F1 | 0.646 |
| Weighted F1 | 0.910 |
| Parameters | ~408,000 |
| Keras size | 1.38 MB |
| TFLite size | 430 KB |
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

If you use these files in your research, please cite:

> Draban, S. (2025). *ECG-TransCovNet: A Hybrid CNN-Transformer for Multi-Class Arrhythmia Detection in ECG Signals*. MSCS Thesis.

## License

MIT — see [LICENSE](../LICENSE) at repository root.
