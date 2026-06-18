# A Unified ML Framework for Fault Detection, Classification, and Self-Healing in Three-Phase Grid-Tied Inverters

**Author:** Md. Atair Rahman Alvi  
**Affiliation:** Department of Electrical and Electronics Engineering, BRAC University, Dhaka, Bangladesh  
**Paper format:** IEEE Conference (IEEEtran)

---

## Abstract

This repository contains the full research code, dataset, and LaTeX paper for a unified machine learning framework that simultaneously performs fault detection, classification, and self-healing in three-phase grid-tied inverters. The framework benchmarks five baseline models and proposes a novel Residual MLP (ResMLP) architecture achieving perfect recall (1.00) on the two most safety-critical fault classes.

---

## Fault Classes

| Class | Fault Type | Description |
|-------|-----------|-------------|
| 0 | Healthy | Balanced three-phase operation |
| 1 | Open-Circuit Switch Fault | IGBT/MOSFET stuck open, half-cycle loss |
| 2 | DC Injection | DC offset in phase current(s) |
| 3 | Grid Voltage Sag/Swell | Amplitude deviation on 1 or 3 phases |
| 4 | Islanding | Loss of grid connection, frequency drift |
| 5 | PLL Desynchronization | Phase angle error between ref and actual |

---

## Results

| Model | Accuracy (%) | F1-Score |
|-------|-------------|----------|
| Random Forest | 99.44 | 0.9945 |
| XGBoost | 98.89 | 0.9890 |
| Shallow MLP | 99.72 | 0.9972 |
| Deep MLP (no skip) | 55.00 | 0.5043 |
| Gradient Boosting | 99.44 | 0.9945 |
| **Proposed ResMLP** | **86.39 (CPU) / 98.06 (GPU val)** | **0.8629** |

> Open-Circuit Fault (Class 1) and Islanding (Class 4): **Recall = 1.00**

---

## Repository Structure

```
grid-tied-inverter-fault-detection/
│
├── README.md
├── paper/
│   ├── main.tex
│   └── figures/
│       ├── fig1.png  ← Model accuracy comparison
│       ├── fig2.png  ← Confusion matrix
│       ├── fig3.png  ← Per-class F1
│       ├── fig4.png  ← Training curves
│       ├── fig5.png  ← Waveform comparison
│       └── fig6.png  ← THD by fault class
├── dataset/
│   ├── inverter_3phase_fault_dataset.csv
│   └── generate_3phase_dataset.py
├── models/
│   └── proposed_only.py
└── results/
    └── results_final.csv
```

---

## Proposed ResMLP Architecture

```
Input (140) → Dense(256) + BN + GELU
            → Res Block 1: Dense(256)×2 + BN×2 + Skip
            → Res Block 2: Dense(256)×2 + BN×2 + Skip
            → Res Block 3: Dense(128)×2 + BN×2 + Skip
            → Res Block 4: Dense(128)×2 + BN×2 + Skip
            → Dense(64) + BN + Dropout(0.15)
            → Dense(6) + Softmax
Total params: 575,046
```

---

## How to Run

### Generate Dataset
```bash
python3 dataset/generate_3phase_dataset.py
```

### Train All Models (Kaggle GPU recommended)
```bash
# Update dataset path in models/proposed_only.py
python3 models/proposed_only.py
```

### Compile Paper
Upload `paper/` to [Overleaf](https://overleaf.com), set compiler to `pdflatex`, compile `main.tex`.

---

## Dependencies

```bash
pip install tensorflow xgboost scikit-learn numpy pandas matplotlib seaborn
```

---

## Reference Papers

1. Baker et al. (COMPEL 2021) — DOI: 10.1109/COMPEL52922.2021.9646062
2. Prabakaran et al. (ICSFT 2026) — DOI: 10.1109/ICSFT66733.2026.11506689

---

## License

MIT License
