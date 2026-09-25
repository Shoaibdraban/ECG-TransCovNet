# Thesis

**Title:** ECG-TransCovNet: A Hybrid CNN-Transformer Architecture for Multi-Class Arrhythmia Detection in ECG Signals

**Author:** Muhammad Shoaib  
**Supervisor:** Dr. Khalid Mehmood  
**University:** Gomal University, Dera Ismail Khan  
**Year:** 2024–2026

---

## 📄 Files

| File | Description | Size |
|------|-------------|------|
| `ECG-TransCovNet_Thesis.pdf` | Complete thesis document | — |

---

## 📝 Abstract

Cardiovascular diseases (CVDs) remain the leading cause of mortality globally, claiming approximately 17.9 million lives annually. In Pakistan, CVDs account for nearly 30% of all deaths, with rural areas facing a critical shortage of cardiologists (< 0.5 per 100,000 population). Automated, accurate, and low-cost electrocardiogram (ECG) diagnosis systems are urgently needed. However, most deep learning models require high-end computational infrastructure unavailable in resource-constrained healthcare settings.

This thesis presents **ECG-TransCovNet**, a hybrid CNN-Transformer architecture for multi-class arrhythmia detection designed for low-power deployment. The research follows a two-phase methodology: (1) synthetic-data-first architectural validation, and (2) real-data clinical validation using the MIT-BIH Arrhythmia Database.

---

## 🎯 Key Results

### Overall Performance (Real MIT-BIH, Locked Test Set)

| Metric | Value |
|--------|-------|
| **Accuracy** | 91.39% |
| **Macro-F1** | 0.646 |
| **Weighted-F1** | 0.910 |
| **Parameters** | ~408,000 |
| **Model Size (Keras)** | 1.38 MB |
| **Model Size (INT8)** | 0.38 MB |
| **CPU Inference** | 5.8 ms |
| **INT8 Inference** | 4.21 ms |

### Per-Class Performance (ECG-TransCovNet)

| Class | Type | Precision | Recall | F1-Score |
|-------|------|-----------|--------|----------|
| N | Normal | 0.969 | 0.945 | 0.957 |
| S | Supraventricular | 0.355 | 0.097 | 0.152 |
| V | Ventricular | 0.781 | 0.939 | 0.853 |
| Q | Paced/Unknown | 0.471 | 0.921 | 0.623 |

### Domain Gap (Synthetic vs Real)

| Metric | Synthetic | Real | Gap |
|--------|-----------|------|-----|
| Accuracy | 100.00% | 91.39% | 8.61 pp |
| Macro-F1 | 1.000 | 0.646 | 0.354 |

### Robustness (Strict Split)

| Split | Accuracy | Macro-F1 |
|-------|----------|----------|
| Standard (DS1/DS2) | 91.39% | 0.646 |
| Strict (202 → train) | 88.15% | 0.632 |
| Difference | 3.24% | 0.014 |

---

## 🏗️ Model Architecture

- **CNN Branch:** 3 conv layers (64→128→256 filters, kernels 7→5→3)
- **Transformer Branch:** 2 encoder layers, 4 attention heads, 128-dim
- **Attention Fusion:** Dynamic weighting (α_cnn + α_trans = 1)
- **RR Branch:** 4 RR-interval features → Dense(32) → Dense(16)
- **Classification Head:** 4 classes (N, S, V, Q)

---

## 📊 Dataset

**MIT-BIH Arrhythmia Database**
- 48 records, 47 subjects
- 109,494 beats
- 360 Hz sampling rate
- MLII single-lead
- 4-class AAMI (N, S, V, Q)

---

## 🎓 Research Contributions

1. **Accuracy-Efficiency Co-Design:** 91.39% accuracy with ~408K parameters (70–300× fewer than comparable models)
2. **Synthetic-to-Real Domain Gap Quantification:** First systematic quantification (8.61 pp accuracy drop)
3. **Subject-Disjoint Evaluation:** Corrected 201/202 same-subject leakage
4. **Holistic Imbalance Mitigation:** Synergistic SMOTE + moderate class weights
5. **Interpretable Attention:** CNN-dominant for V-class, Transformer-dominant for Q-class
6. **Deployment-Ready:** 0.38 MB INT8 model for resource-constrained settings

---

## 📚 Citation

If you use this thesis in your research, please cite:

@mastersthesis{shoaib2026ecg,
  title={ECG-TransCovNet: A Hybrid CNN-Transformer Architecture for Multi-Class Arrhythmia Detection in ECG Signals},
  author={Shoaib, Muhammad},
  year={2026},
  school={Gomal University, Dera Ismail Khan}
}

---

## 🔗 Related Resources

- **Code Repository:** [github.com/Shoaibdraban/ECG-TransCovNet](https://github.com/Shoaibdraban/ECG-TransCovNet)
- **Portfolio:** [shoaibdraban-portfolio.netlify.app](https://shoaibdraban-portfolio.netlify.app)
- **Contact:** [shoaibdraban@gmail.com](mailto:shoaibdraban@gmail.com)

---

## 📧 Contact

**Muhammad Shoaib**  
Email: [shoaibdraban@gmail.com](mailto:shoaibdraban@gmail.com)  
GitHub: [@Shoaibdraban](https://github.com/Shoaibdraban)  
Portfolio: [shoaibdraban-portfolio.netlify.app](https://shoaibdraban-portfolio.netlify.app)

---

## 📜 License

This thesis document is © 2026 Muhammad Shoaib. All Rights Reserved.

The thesis PDF is provided for academic reference only. Please cite appropriately if used in research.
