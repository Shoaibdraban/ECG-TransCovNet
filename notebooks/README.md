# Jupyter Notebooks

This folder contains the Jupyter notebooks used in the thesis.

---

## Notebooks

| Notebook | Description |
|----------|-------------|
| Phase0_Audit.ipynb | Methodological audit and corrections |
| Phase1_Synthetic.ipynb | Synthetic data generation and validation |
| Phase2_Real_MITBIH.ipynb | Real MIT-BIH data training and evaluation |
| Phase3_Deployment.ipynb | TensorFlow Lite conversion and deployment |

---

## Phase 0: Audit

Documents and corrects five methodological flaws in the original thesis:

1. Subject-level leakage (201/202 same-subject exception)
2. Silent merging of S-class into N-class
3. Hyperparameter tuning on test set
4. Hard-coded, non-reproducible results
5. Incomplete training reproducibility

---

## Phase 1: Synthetic Data

- 50,000 heartbeat segments
- 5 AAMI classes (N, S, V, F, Q)
- ECGsim library (McSharry et al., 2003)
- 100% architectural validation accuracy

---

## Phase 2: Real MIT-BIH Data

- 48 records, 47 subjects, 109,494 beats
- 4-class AAMI (N, S, V, Q)
- Subject-disjoint split
- 91.39% accuracy, 0.646 macro-F1

---

## Phase 3: Deployment

- TensorFlow Lite conversion
- FP16 and INT8 quantization
- 0.38 MB final model size
- 4.21 ms INT8 inference

---

## Requirements

pip install jupyter tensorflow numpy scipy scikit-learn matplotlib

---

## Reproducibility

All notebooks use:
- Random seed: 42
- Fixed hyperparameters
- Documented experimental conditions
- Phase 0 to 6 corrective pipeline

---

## Key Highlights

- Two-phase methodology (synthetic to real)
- Subject-disjoint split (201/202 correction)
- Capped SMOTE (25% of majority class)
- Moderate class weights (sqrt balanced)
- 5-fold GroupKFold cross-validation
- INT8 quantization for edge deployment

---

## Citation

If you use these notebooks in your research, please cite:

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
- Results: ../results
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

These notebooks are part of the ECG-TransCovNet project and are licensed under the MIT License — see the LICENSE file for details.
