# Manuscript release (v0.9-manuscript)

This tagged version contains the **seven-view classifier used in the manuscript**: "Validation and Performance of Two Open-Source Comprehensive Cardiology AI Echocardiography Models on Cardiac POCUS for Pericardial Effusion and Left Ventricular Function". It was used to select view-qualifying clips: PSLA, PSSA, A4C or SX with confidence ≥0.90.

- **Weights:** `weights/ed_view_convnext_fp16.pt`. ConvNeXt-Tiny with seven classes (PSLA, PSSA, A4C, SX, IVC, Lung, Indeterminate). Trained without horizontal-flip augmentation. Stored in half precision.
- **Code:** `predict.py`, `train.py` and `build_dataset.py`, exactly as in commit `7f014e0`.
- **Held-out test performance:** 2,055 loops from 259 unseen patients. Accuracy 0.840, Cohen's κ 0.810; accuracy 0.88 across the four cardiac views.
- **Reproducibility check:** the fp16 weights were run with `predict.py` on 202 cohort clips from 30 studies. They reproduced the study's view label and ≥0.90 qualification for every clip; the largest confidence difference was 0.004.

The `main` branch holds the later v1.0.0 model. That model was retrained with PSLA/IVC flip augmentation and adds a preliminary orientation output. **It was not used in the manuscript.**

Research use only. Not a medical device.
