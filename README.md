# ED-Native Echo View + Non-Cardiac Gate Classifier

A view classifier for **emergency-department point-of-care echocardiography (ED-POCUS)** that labels
each ultrasound loop as one of:

```
PSLA · PSSA · A4C · SX · IVC · Lung · Indeterminate
```

Unlike cardiology-trained view classifiers, it (1) works in the **native ED probe-indicator
orientation** (no horizontal-flip correction needed), and (2) includes an explicit **Lung /
Indeterminate gate** so non-cardiac loops can be triaged *before* they reach a downstream model
(PanEcho, EchoPrime, etc.) that would otherwise aggregate over them.

> **Research use only. Not a medical device.** Outputs require clinician review and are not a
> diagnostic determination.

## Performance (held-out test: 2,055 loops, 259 unseen patients)

| Metric | This model | EchoPrime (off-the-shelf, best-of-both) |
|---|---:|---:|
| Accuracy | **0.840** | 0.607 |
| Cohen's κ | **0.810** | 0.545 |

κ 0.810 is at the expert–expert reliability ceiling (κ 0.835). Per-class recall (ours vs EchoPrime):
**A4C 0.909 vs 0.390**, **Lung 0.851 vs 0.000** (EchoPrime has no Lung class), PSLA 0.877 (no flip
needed), PSSA 0.877, SX 0.842, IVC 0.870. The `Indeterminate` class is intentionally a low-confidence
"reject" bucket (recall 0.35; it is intrinsically ambiguous — expert–expert κ ≈ 0.28 there).

## Install

```bash
pip install torch torchvision opencv-python numpy        # + scikit-learn only if you retrain
```

Runs on CPU, CUDA, or Apple-silicon MPS (auto-detected). The model is a ConvNeXt-Tiny
(~28M params); shipped weights are fp16 (~56 MB).

## Predict

```bash
# a folder of .mp4 loops:
python predict.py --input path/to/clips --out predictions.csv

# a tree of study folders (recurses for *.mp4):
python predict.py --input path/to/studies --out predictions.csv

# a single loop:
python predict.py --input clip.mp4
```

`predictions.csv` has one row per loop: `path, view, confidence, is_cardiac, view_2, prob_2`.
Loops with `is_cardiac = False` (Lung / Indeterminate) are the ones to gate out.

## Test it on your own reviewed clips

1. Put your `.mp4` loops in a folder.
2. (Optional) make a `labels.csv` with two columns, `filename,view`, using the label set above.
3. Run:

```bash
python predict.py --input your_clips/ --labels labels.csv --out your_predictions.csv
```

It prints accuracy and a confusion matrix vs. your labels, and writes per-loop predictions.

## How it was trained (code included; data not distributed)

- **Labels:** expert clip-level view annotations from a blinded ED-POCUS re-review (13,801 loops,
  7 classes incl. Lung/Indeterminate), modal label per loop.
- **Input:** first frame per loop, aspect-preserving crop to 224×224.
- **Split:** by patient, 70/15/15 (no study straddles splits).
- **Model/opt:** ImageNet ConvNeXt-Tiny fine-tuned, AdamW (lr 1e-4, wd 0.05), cosine schedule,
  12 epochs, class-weighted cross-entropy. **No horizontal-flip augmentation** (learns ED
  orientation natively). Best-validation checkpoint retained.

`build_dataset.py` and `train.py` are included for transparency; they expect the labeled loops,
which are **not** distributed (patient data under IRB). See each script's header for the expected
inputs.

## Intended use & limitations

- ED-POCUS, first-frame view triage. Single-center training data; **external validation pending.**
- First-frame (not full-clip) input; multi-frame input is a natural extension.
- `Indeterminate` is a reject bucket, not a reliable target class.
- Not validated for, and not intended for, autonomous clinical decision-making.

## Data & ethics

Trained on de-identified ED-POCUS loops under a Yale IRB-exempt determination. **No patient media or
labels are distributed** — only the trained model weights and code. Confirm your own institutional
permissions before deploying on clinical data.

## Citation

Moore C, et al. *An ED-native echocardiographic view and non-cardiac gate classifier.* (in
preparation), 2026.

## License

See `LICENSE`.
