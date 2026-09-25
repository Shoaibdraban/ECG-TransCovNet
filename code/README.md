# Source Code

This folder contains the complete source code for ECG-TransCovNet — a Hybrid CNN-Transformer Architecture for Multi-Class Arrhythmia Detection in ECG Signals.

---

## Files

| File | Description |
|------|-------------|
| preprocessing.py | Data preprocessing pipeline (filtering, R-peak detection, segmentation, normalization) |
| model.py | CNN-Transformer architecture definition (ECG-TransCovNet) |
| train.py | Training script with two-phase methodology (synthetic to real) |
| evaluate.py | Evaluation script (metrics, confusion matrix, ROC curves) |
| requirements.txt | Python dependencies |

---

## Requirements

### System Requirements

- Python 3.10+
- TensorFlow 2.15+
- 8 GB RAM minimum (16 GB recommended)
- CPU or GPU (NVIDIA recommended for training)

### Python Dependencies

pip install -r requirements.txt

requirements.txt contents:

tensorflow==2.15.0
keras==2.15.0
numpy==1.24.3
scipy==1.11.4
scikit-learn==1.3.2
matplotlib==3.7.2
seaborn==0.12.2
pandas==2.0.3
imbalanced-learn==0.11.0
wfdb==4.1.2

---

## Usage

### 1. Preprocessing

python preprocessing.py --data_dir ./data --output_dir ./processed

What it does:
- Baseline wander removal (median filter, 200ms window)
- Powerline notch (50Hz) + bandpass (0.5 to 40Hz Butterworth)
- Pan-Tompkins R-peak detection
- Beat segmentation (259 samples: 99 before + 160 after R-peak)
- Per-beat Z-score normalization
- RR-interval feature extraction (4 features)
- Capped SMOTE balancing (25% of majority class)

### 2. Training

python train.py --epochs 30 --batch_size 128 --lr 5e-4

Two-Phase Training Strategy:

| Phase | Data | Loss Function | Optimizer | Epochs |
|-------|------|---------------|-----------|--------|
| Phase 1 | Synthetic (50,000 beats, 5 classes) | Focal Loss (gamma=2.0) | Adam (lr=1e-3) | 100 |
| Phase 2 | Real MIT-BIH (109,494 beats, 4 classes) | Sparse Categorical Cross-Entropy + sqrt class weights | Adam (lr=5e-4) | 30 |

Training Features:
- 5-fold GroupKFold cross-validation by subject
- Early stopping (patience 6 to 15)
- ReduceLROnPlateau scheduler
- Random seed: 42 (reproducibility)

### 3. Evaluation

python evaluate.py --model_path ./model/ecg_transcovnet.keras

Metrics Reported:
- Accuracy
- Macro-F1 / Weighted-F1
- Per-class Precision, Recall, F1-Score
- AUROC / AUPRC
- Confusion Matrix
- Cohen's Kappa

### 4. TensorFlow Lite Conversion

import tensorflow as tf

model = tf.keras.models.load_model('ecg_transcovnet.keras')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
tflite_model = converter.convert()

with open('ecg_transcovnet_int8.tflite', 'wb') as f:
    f.write(tflite_model)

---

## Model Architecture

### CNN Branch (Local Feature Extraction)

| Layer | Filters | Kernel | Output Shape |
|-------|---------|--------|--------------|
| Conv1D + BatchNorm + ReLU + MaxPool | 64 | 7 | (129, 64) |
| Conv1D + BatchNorm + ReLU + MaxPool | 128 | 5 | (64, 128) |
| Conv1D + BatchNorm + ReLU + GlobalAvgPool | 256 | 3 | (256,) |

### Transformer Branch (Global Context Modeling)

| Component | Details |
|-----------|---------|
| Linear Projection | 259 to 128 dimensions |
| Positional Encoding | Learned embeddings |
| Encoder Layers | 2 layers |
| Attention Heads | 4 heads |
| Feed-Forward Dim | 512 |
| Dropout | 0.1 |
| Output | 128-dim vector |

### Attention-Based Fusion

F_concat = [F_cnn; F_trans]                        # 384-dim
alpha = softmax(W * F_concat + b)                  # 2-dim
F_fused = alpha_cnn * F_cnn + alpha_trans * F_trans   # 256-dim

### RR Branch

| Layer | Units |
|-------|-------|
| Input | 4 RR-interval features |
| Dense 1 | 32 |
| Dense 2 | 16 |

### Classification Head

| Layer | Units | Activation |
|-------|-------|------------|
| Dense | 64 | ReLU |
| Dropout | — | 0.4 |
| Output Dense | 4 | Softmax |

---

## Dataset

### MIT-BIH Arrhythmia Database

| Aspect | Value |
|--------|-------|
| Source | PhysioNet |
| Records | 48 |
| Subjects | 47 (201 and 202 same subject) |
| Total Beats | 109,494 |
| Sampling Rate | 360 Hz |
| Lead | MLII (single-lead) |
| Classes | 4-class AAMI (N, S, V, Q) |

### AAMI Class Mapping

| Class | Type | Description |
|-------|------|-------------|
| N | Normal | Normal sinus rhythm, bundle branch blocks |
| S | Supraventricular | Atrial premature beats, SVT, AFib |
| V | Ventricular | PVCs, ventricular tachycardia |
| Q | Paced/Unknown | Paced beats, unclassifiable beats |

### Data Split

| Split Protocol | Train | Validation | Test |
|----------------|-------|------------|------|
| Standard (DS1/DS2) | 20 records | 5 records | 23 records |
| Strict (Subject-Disjoint) | 21 records | 5 records | 22 records |

Note: Strict split moves record 202 to training (same subject as 201) to eliminate subject-level leakage.

---

## Results

### Overall Performance (Real MIT-BIH)

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|-------|----------|----------|-------------|
| Baseline A (CNN + RR) | 85.29% | 0.609 | 0.870 |
| ECG-TransCovNet (Full) | 91.39% | 0.646 | 0.910 |
| Ensemble (w=0.5) | 90.84% | 0.658 | 0.908 |

### Per-Class Performance

| Class | Type | Precision | Recall | F1-Score | AUROC | AUPRC |
|-------|------|-----------|--------|----------|-------|-------|
| N | Normal | 0.969 | 0.945 | 0.957 | 0.958 | 0.992 |
| S | Supraventricular | 0.355 | 0.097 | 0.152 | 0.861 | 0.207 |
| V | Ventricular | 0.781 | 0.939 | 0.853 | 0.986 | 0.954 |
| Q | Paced/Unknown | 0.471 | 0.921 | 0.623 | 0.990 | 0.899 |

### Computational Efficiency

| Metric | Value |
|--------|-------|
| Parameters | ~408,000 |
| Model Size (Keras) | 1.38 MB |
| Model Size (INT8) | 0.38 MB |
| CPU Inference | 5.8 ms |
| INT8 Inference | 4.21 ms |
| Raspberry Pi 4 | 18.7 ms/beat |

### Ablation Study (Class Imbalance)

| Configuration | Macro-F1 |
|---------------|----------|
| Baseline (no mitigation) | 0.35 |
| SMOTE only | 0.48 |
| Class weights only | 0.45 |
| SMOTE + Class weights | 0.55 |

### Attention Weights by Class

| Class | alpha_cnn | alpha_trans | Dominant Branch |
|-------|-----------|-------------|-----------------|
| N | 0.52 | 0.48 | Balanced |
| S | 0.48 | 0.52 | Balanced |
| V | 0.71 | 0.29 | CNN |
| Q | 0.32 | 0.68 | Transformer |

---

## Reproducibility

### Fixed Parameters

| Parameter | Value |
|-----------|-------|
| Random Seed | 42 |
| NumPy Seed | 42 |
| TensorFlow Seed | 42 |
| Python Hash Seed | 42 |

### Environment

| Component | Version |
|-----------|---------|
| OS | Ubuntu 22.04 LTS |
| Python | 3.10.12 |
| TensorFlow | 2.15.0 |
| Keras | 2.15.0 |
| NumPy | 1.24.3 |
| SciPy | 1.11.4 |
| Scikit-learn | 1.3.2 |

### Corrective Pipeline (Phase 0 to 6)

The original thesis had five methodological flaws that were corrected in this work:

1. Subject-level leakage: Record 201 (train) and 202 (test) from same subject — fixed
2. Silent S-class merge: S-class was merged into N-class without documentation — fixed with explicit 4-class AAMI
3. Test set tuning: Record 228 used for Pan-Tompkins tuning — fixed
4. Hard-coded results: Non-reproducible values — fixed with seed 42
5. Incomplete reproducibility: Training pipeline incomplete — fixed with documented notebooks

All corrections are documented in the notebooks/ folder.

---

## Citation

If you use this code in your research, please cite:

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
- Model Files: ../model
- Portfolio: https://shoaibdraban-portfolio.netlify.app

---

## Contact

Muhammad Shoaib
whatsapp: +923467851061
Email: shoaibdraban@gmail.com
GitHub: https://github.com/Shoaibdraban
Portfolio: https://shoaibdraban-portfolio.netlify.app

---

## License

This code is licensed under the MIT License — see the LICENSE file for details.
