# Distribution Shift and Spurious Correlations in Deep Learning

This project investigates how deep learning models can learn spurious correlations instead of meaningful features, leading to failures under distribution shift.

---

## Motivation

High accuracy on training data does not always reflect true understanding.  
Models often exploit shortcuts present in the data, which can break when the data distribution changes.

This project explores how such behavior emerges and how it depends on both **bias strength** and **model capacity**.

---

## Methodology

- Introduced synthetic bias into CIFAR-10 training data  
- Controlled bias strength: **0.0, 0.2, 0.5, 1.0**  
- Trained two models:
  - Small CNN  
  - Larger CNN  
- Evaluated performance on unbiased test data  

---

## Results

- Increasing bias strength leads to a clear drop in test accuracy  
- Models rely on spurious correlations (color bias)  
- Larger models show higher sensitivity to bias  

### Visualization

![Results](final_shift_results.png)

---

## Key Insight

Models can achieve high performance for the wrong reasons.  
Understanding distribution shift is essential for building reliable AI systems.

---

## Run

```bash
pip install torch torchvision numpy matplotlib
python3 shift_full.py
