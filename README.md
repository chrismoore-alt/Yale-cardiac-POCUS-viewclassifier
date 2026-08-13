# ED Echo View + Orientation + Non-Cardiac Gate Classifier

A point-of-care echocardiography classifier for the **emergency department (ED)** that, for each
ultrasound loop, reports:

- **view** — `PSLA · PSSA · A4C · SX · IVC · Lung · Indeterminate` (orientation-invariant)
- **is_cardiac** — a **gate**: `True` for the five cardiac views, `False` for Lung / Indeterminate
- **orientation** (PSLA/IVC only) — `EM` (emergency-medicine indicator-left) vs `cards` (cardiology
  indicator-right), so a loop can be flipped to the cardiology convention before it is fed to a
  cardiology-trained model (PanEcho, EchoPrime, …)
- **action** — `gate_out` | `flip_then_use` | `use`

Unlike cardiology-trained view classifiers, it (1) works in the **native ED orientation**, (2) has an
explicit **non-cardiac gate** so lung/off-target loops can be triaged out *before* a downstream model
aggregates over them, and (3) **identifies and can standardize orientation** across ED conventions.

> **Research use only. Not a medical device.** Outputs require clinician review.

## Performance (held-out test: unseen patients)

**View** (orientation-invariant): accuracy 0.869, **Cohen's κ 0.838** — at the expert–expert
reliability ceiling (κ 0.835), and far above off-the-shelf EchoPrime on the same task (κ 0.545).
Per-view recall: PSLA 0.95, PSSA 0.89, A4C 0.89, SX 0.80, IVC 0.87, **Lung (gate) 0.88**;
Indeterminate 0.37 (intentionally a low-confidence "reject" bucket).

**Orientation** (PSLA/IVC) — **PRELIMINARY, do not rely on yet.** On synthetic horizontal flips of
ED clips the orientation head scores ~0.999 and 98.1% convention-invariance — but these numbers
**overstate real-world performance**: on an out-of-distribution study from a *different machine*
(Philips Compact 5500 vs. the Sparq used in training) the head confidently mislabeled a
clearly-EM IVC as cardiology. The head appears to have learned synthetic-flip / machine-specific
cues rather than true anatomical left/right. Orientation detection therefore requires **real
cross-convention, multi-machine data** before it can be reported as validated. (View classification
and the gate, by contrast, generalized to that other machine.)

## Install

```bash
pip install torch torchvision opencv-python numpy        # + scikit-learn only if you retrain
```

Runs on CPU, CUDA, or Apple-silicon MPS (auto-detected). ConvNeXt-Tiny backbone; shipped weights are
fp16 (~56 MB) at `weights/ed_view_orient_fp16.pt`.

## Predict

```bash
python predict.py --input path/to/clips --out predictions.csv    # folder of .mp4 (recurses)
python predict.py --input clip.mp4                               # single loop
```

`predictions.csv` columns: `path, view, confidence, is_cardiac, orientation, action, view_2, prob_2`.

**Actions:** `gate_out` (non-cardiac/uncertain → drop from cardiac analysis) · `flip_then_use` (cardiac
PSLA/IVC in EM orientation → horizontally flip before a cardiology-trained model) · `use`.

Example (one study):
```
clip_0001  PSLA           0.999  cardiac  EM   flip_then_use
clip_0003  PSSA           0.998  cardiac  --   use
clip_0005  Lung           1.000  non-card --   gate_out
clip_0007  A4C            1.000  cardiac  --   use
```

## Test it on your own reviewed clips

1. Put your `.mp4` loops in a folder.
2. (Optional) `labels.csv` with columns `filename,view` in the 7-view vocab above.
3. Run `python predict.py --input your_clips/ --labels labels.csv --out results.csv` — it prints view
   accuracy + a confusion matrix vs. your reads, and writes per-loop predictions.

## How it was trained (code included; data not distributed)

- **Labels:** expert clip-level view annotations from a blinded ED-POCUS re-review (13,801 loops).
- **Input:** first frame, aspect-preserving crop to 224×224. **Split by patient** (70/15/15).
- **Orientation via selective flipping:** because only **PSLA and IVC** differ between EM and
  cardiology conventions (PSSA/A4C/SX are congruent under the "double reverse"), training flips **only
  PSLA/IVC** loops to synthesize cardiology-orientation examples — yielding a 9-way label space
  {PSLA_EM, PSLA_cards, IVC_EM, IVC_cards, PSSA, A4C, SX, Lung, Indeterminate} that collapses to the
  7 views plus a PSLA/IVC orientation.
- **Model/opt:** ImageNet ConvNeXt-Tiny fine-tuned; AdamW (1e-4, wd 0.05); cosine schedule; 12 epochs;
  class-weighted cross-entropy; **no whole-image flip augmentation** for the congruent views.

`build_dataset.py`, `build_orient_dataset.py`, and `train.py` are included for transparency; they
expect the labeled loops, which are **not** distributed (patient data under IRB).

## Intended use & limitations

- ED-POCUS, first-frame view triage + non-cardiac gate. Single-center training data; **external
  validation pending.** View + gate generalized to a second machine (Philips Compact 5500) in
  informal testing; **orientation detection did not** and is preliminary (see Performance).
- **Preprocessing is format-bound.** Trained on the center-crop of QPath-style ~800×600 exports.
  Different export layouts/aspect ratios may reduce reliability, and applying a different crop (e.g.
  ultrasound-sector detection) *at inference only* degrades accuracy (train/inference mismatch) —
  robust multi-format support needs a sector-consistent retrain.
- `Indeterminate` is a reject bucket, not a reliable target class.
- Not validated for, and not intended for, autonomous clinical decision-making.

## Data & ethics

Trained on de-identified ED-POCUS loops under a Yale IRB-exempt determination. **No patient media or
labels are distributed** — only trained weights and code.

## Citation

Moore C, et al. *An ED-native echocardiographic view, orientation, and non-cardiac gate classifier.*
(in preparation), 2026.

## License

See `LICENSE`.
