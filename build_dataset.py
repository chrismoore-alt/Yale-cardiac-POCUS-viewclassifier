# NOTE: builds the labeled first-frame dataset from expert view labels + local loops.
# Expects: expert_view_labels_long.csv (case_id,clip_filename,view_label) and a clips tree.
# Data is not distributed (patient data under IRB). Adapt paths to reuse.

#!/usr/bin/env python3
# Build the ED view-classifier training set from expert clip-level view labels.
# For each expert-labeled clip: extract first frame -> crop_and_scale 224x224 (uint8, RGB).
# Patient-level train/val/test split (no study straddles splits).
import os, sys, csv, collections, glob
import numpy as np
import cv2

# crop_and_scale — inlined from EchoPrime/utils/utils.py (byte-identical preprocessing)
def crop_and_scale(img, res=(224, 224), interpolation=cv2.INTER_CUBIC, zoom=0.1):
    in_res = (img.shape[1], img.shape[0])
    r_in = in_res[0] / in_res[1]; r_out = res[0] / res[1]
    if r_in > r_out:
        pad = int(round((in_res[0] - r_out * in_res[1]) / 2)); img = img[:, pad:-pad]
    if r_in < r_out:
        pad = int(round((in_res[1] - in_res[0] / r_out) / 2)); img = img[pad:-pad]
    if zoom != 0:
        px = round(int(img.shape[1] * zoom)); py = round(int(img.shape[0] * zoom))
        img = img[py:-py, px:-px]
    return cv2.resize(img, res, interpolation=interpolation)

OUT = "/path/to/ai_echo_local"
LAB = f"{OUT}/outputs/expert_labels/expert_view_labels_long.csv"
DL  = f"{OUT}/downloads"
DS  = f"{OUT}/view_ds"
os.makedirs(DS, exist_ok=True)

CLASSES = ["PSLA", "PSSA", "A4C", "SX", "IVC", "Lung", "Indeterminate"]
CIDX = {c: i for i, c in enumerate(CLASSES)}

# modal expert label per (case, clip)
raw = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
for r in csv.DictReader(open(LAB)):
    raw[r["case_id"]][r["clip_filename"]][r["view_label"]] += 1
items = []  # (case_id, clip_filename, label_idx)
for cid, clips in raw.items():
    for cl, cnt in clips.items():
        lab = cnt.most_common(1)[0][0]
        if lab in CIDX:
            items.append((cid, cl, CIDX[lab]))
print(f"labeled clips (modal): {len(items)}")

def first_frame(path):
    cap = cv2.VideoCapture(path); ok, fr = cap.read(); cap.release()
    if not ok: return None
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)

X, y, cases = [], [], []
miss = 0
for i, (cid, cl, lab) in enumerate(items):
    p = os.path.join(DL, cid, "panecho_input", cl)
    if not os.path.exists(p): miss += 1; continue
    fr = first_frame(p)
    if fr is None: miss += 1; continue
    X.append(crop_and_scale(fr).astype(np.uint8)); y.append(lab); cases.append(cid)
    if (i + 1) % 2000 == 0: print(f"  {i+1}/{len(items)} processed ({miss} missing)")

X = np.stack(X); y = np.array(y, np.int64); cases = np.array(cases)
print(f"dataset: X={X.shape} y={y.shape}  missing/unreadable={miss}")
print("class counts:", {CLASSES[i]: int((y == i).sum()) for i in range(len(CLASSES))})

# patient-level split 70/15/15 (deterministic hash of case_id)
def bucket(cid):
    h = int(__import__("hashlib").md5(cid.encode()).hexdigest(), 16) % 100
    return "train" if h < 70 else ("val" if h < 85 else "test")
split = np.array([bucket(c) for c in cases])
for s in ["train", "val", "test"]:
    m = split == s
    print(f"  {s}: {m.sum()} clips, {len(set(cases[m]))} cases")

np.savez_compressed(f"{DS}/view_dataset.npz", X=X, y=y, cases=cases, split=split,
                    classes=np.array(CLASSES))
print(f"saved -> {DS}/view_dataset.npz ({os.path.getsize(f'{DS}/view_dataset.npz')/1e6:.0f} MB)")
