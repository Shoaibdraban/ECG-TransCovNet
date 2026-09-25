# Results

This folder contains the experimental results and plots for ECG-TransCovNet.

---

## Files

| File | Description |
|------|-------------|
| confusion_matrix.png | Normalized confusion matrix (Real MIT-BIH) |
| per_class_performance.png | Per-class Precision, Recall, F1-Score |
| training_curves.png | Training and validation accuracy curves |
| roc_curves.png | ROC curves for all 4 classes |

---

## Key Results

### Overall Performance (Real MIT-BIH, Locked Test Set)

| Model | Accuracy | Macro-F1 | Weighted-F1 |
|-------|----------|----------|-------------|
| Baseline A (CNN + RR) | 85.29% | 0.609 | 0.870 |
| ECG-TransCovNet (Full) | 91.39% | 0.646 | 0.910 |
| Ensemble (w=0.5) | 90.84% | 0.658 | 0.908 |

### Per-Class Performance (ECG-TransCovNet)

| Class | Type | Precision | Recall | F1-Score | AUROC | AUPRC |
|-------|------|-----------|--------|----------|-------|-------|
| N | Normal | 0.969 | 0.945 | 0.957 | 0.958 | 0.992 |
| S | Supraventricular | 0.355 | 0.097 | 0.152 | 0.861 | 0.207 |
| V | Ventricular | 0.781 | 0.939 | 0.853 | 0.986 | 0.954 |
| Q | Paced/Unknown | 0.471 | 0.921 | 0.623 | 0.990 | 0.899 |

### Domain Gap (Synthetic vs Real)

| Metric | Synthetic | Real | Gap |
|--------|-----------|------|-----|
| Accuracy | 100.00% | 91.39% | 8.61 pp |
| Macro-F1 | 1.000 | 0.646 | 0.354 |

### Robustness (Strict Split)

| Split | Accuracy | Macro-F1 |
|-------|----------|----------|
| Standard (DS1/DS2) | 91.39% | 0.646 |
| Strict (202 to train) | 88.15% | 0.632 |
| Difference | 3.24% | 0.014 |

---

## Efficiency

| Metric | Value |
|--------|-------|
| Parameters | ~408,000 |
| Model Size (Keras) | 1.38 MB |
| Model Size (INT8) | 0.38 MB |
| CPU Inference | 5.8 ms |
| INT8 Inference | 4.21 ms |
| Raspberry Pi 4 | 18.7 ms/beat |

---

## TensorFlow Lite Optimization

| Format | Model Size | Size Reduction | CPU Inference |
|--------|------------|----------------|---------------|
| Original Keras | 1.38 MB | — | 5.80 ms |
| TFLite (FP32) | 1.38 MB | 0% | 5.21 ms |
| TFLite (FP16) | 0.72 MB | 47.8% | 4.89 ms |
| TFLite (INT8) | 0.38 MB | 72.5% | 4.21 ms |

---

## End-to-End Inference Pipeline (Raspberry Pi 4)

| Pipeline Stage | Time (ms) |
|----------------|-----------|
| Signal acquisition (10-sec ECG) | 0.5 |
| Noise filtering | 1.2 |
| R-peak detection (Pan-Tompkins) | 3.5 |
| Heartbeat segmentation | 0.8 |
| Per-beat normalization | 0.3 |
| Model inference (per beat) | 18.7 |
| Post-processing (class assignment) | 0.5 |
| Total per beat | 25.5 |

---

## Confusion Matrix Summary

### Real MIT-BIH Data

| Actual / Predicted | N | S | V | Q |
|--------------------|-----|------|------|------|
| N | 94.5% | 8.9% | 9.0% | 56.0% |
| S | 34.0% | 9.7% | 19.0% | 41.3% |
| V | 16.1% | 4.5% | 93.9% | 4.4% |
| Q | 13.5% | 0.4% | 0.8% | 92.1% |

---

## Ablation Study (Class Imbalance)

| Configuration | S-class F1 | V-class F1 | Q-class F1 | Macro-F1 |
|---------------|-----------|-----------|-----------|----------|
| Baseline (no mitigation) | 0.08 | 0.42 | 0.28 | 0.35 |
| SMOTE only (25% cap) | 0.12 | 0.58 | 0.45 | 0.48 |
| Class weights only (sqrt) | 0.10 | 0.55 | 0.42 | 0.45 |
| SMOTE + Class weights | 0.15 | 0.62 | 0.52 | 0.55 |

---

## Attention Weights by Class

| Class | alpha_cnn | alpha_trans | Dominant Branch |
|-------|-----------|-------------|-----------------|
| N | 0.52 | 0.48 | Balanced |
| S | 0.48 | 0.52 | Balanced |
| V | 0.71 | 0.29 | CNN |
| Q | 0.32 | 0.68 | Transformer |

---

## Key Findings

1. ECG-TransCovNet achieves 91.39% accuracy on real MIT-BIH data
2. 6.1 percentage point improvement over Baseline A (CNN + RR only)
3. V-class F1 = 0.853 (clinically critical class)
4. Q-class F1 = 0.623 (high recall of 92.1%)
5. S-class remains challenging (F1 = 0.152) due to single-lead constraints
6. Domain gap: 8.61 percentage points (synthetic to real)
7. INT8 quantization reduces model size by 72.5% with no accuracy loss
8. Attention weights are clinically interpretable (CNN for morphology, Transformer for rhythm)

---

## Citation

If you use these results in your research, please cite:

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
Email: shoaibdraban@gmail.com
WhatsApp: +92 346 7851061
GitHub: https://github.com/Shoaibdraban
Portfolio: https://shoaibdraban-portfolio.netlify.app

---

## License

This results documentation is part of the ECG-TransCovNet project and is licensed under the MIT License — see the LICENSE file for details.
