#!/usr/bin/env python3
"""
ED-native echocardiographic view + non-cardiac gate classifier — inference.

Classifies each ultrasound loop (mp4) into one of:
  PSLA, PSSA, A4C, SX, IVC, Lung, Indeterminate
in the emergency-department probe-indicator convention (no orientation flip needed).
'Lung' and 'Indeterminate' act as a non-cardiac / uncertain GATE.

Usage:
  python predict.py --input path/to/clips            # a folder of .mp4 files
  python predict.py --input path/to/study_folders    # folders each containing loops (recursive)
  python predict.py --input clip.mp4                  # a single loop
  python predict.py --input clips/ --labels my_labels.csv   # + agreement vs your labels

Outputs a CSV (default predictions.csv) with, per loop:
  path, view (argmax), confidence, is_cardiac (gate), top2 view/prob
With --labels (csv columns: filename,view) it also prints accuracy and a confusion matrix.
"""
import argparse, csv, glob, os, sys
import numpy as np
import cv2
import torch
import torchvision

CLASSES = ["PSLA", "PSSA", "A4C", "SX", "IVC", "Lung", "Indeterminate"]
CARDIAC = {"PSLA", "PSSA", "A4C", "SX", "IVC"}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def crop_and_scale(img, res=(224, 224), interpolation=cv2.INTER_CUBIC, zoom=0.1):
    in_res = (img.shape[1], img.shape[0]); r_in = in_res[0] / in_res[1]; r_out = res[0] / res[1]
    if r_in > r_out:
        pad = int(round((in_res[0] - r_out * in_res[1]) / 2)); img = img[:, pad:-pad]
    if r_in < r_out:
        pad = int(round((in_res[1] - in_res[0] / r_out) / 2)); img = img[pad:-pad]
    if zoom:
        px = int(img.shape[1] * zoom); py = int(img.shape[0] * zoom); img = img[py:-py, px:-px]
    return cv2.resize(img, res, interpolation=interpolation)


def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")


def load_model(weights, device):
    ck = torch.load(weights, map_location="cpu")
    m = torchvision.models.convnext_tiny()
    m.classifier[-1] = torch.nn.Linear(m.classifier[-1].in_features, len(CLASSES))
    sd = {k: v.float() for k, v in ck["model"].items()}   # fp16 weights -> float for inference
    m.load_state_dict(sd)
    m.eval().to(device)
    return m


def first_frame(path):
    cap = cv2.VideoCapture(str(path)); ok, fr = cap.read(); cap.release()
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB) if ok else None


def find_clips(inp):
    if os.path.isfile(inp) and inp.lower().endswith(".mp4"):
        return [inp]
    return sorted(glob.glob(os.path.join(inp, "**", "*.mp4"), recursive=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="mp4 file, folder of mp4s, or tree of study folders")
    ap.add_argument("--weights", default=os.path.join(os.path.dirname(__file__), "weights", "ed_view_convnext_fp16.pt"))
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--labels", default=None, help="optional CSV with columns: filename,view")
    args = ap.parse_args()

    device = get_device(); print(f"device: {device}")
    model = load_model(args.weights, device)
    clips = find_clips(args.input)
    print(f"clips found: {len(clips)}")
    if not clips: sys.exit("no .mp4 clips found under --input")

    rows = []
    with torch.no_grad():
        for p in clips:
            fr = first_frame(p)
            if fr is None:
                rows.append({"path": p, "view": "READ_ERROR", "confidence": 0, "is_cardiac": "",
                             "view_2": "", "prob_2": ""}); continue
            x = torch.from_numpy(crop_and_scale(fr)).permute(2, 0, 1).float() / 255.0
            x = ((x - MEAN) / STD).unsqueeze(0).to(device)
            prob = torch.softmax(model(x), dim=1)[0].cpu().numpy()
            order = np.argsort(-prob)
            view = CLASSES[order[0]]
            rows.append({"path": p, "view": view, "confidence": round(float(prob[order[0]]), 3),
                         "is_cardiac": (view in CARDIAC), "view_2": CLASSES[order[1]],
                         "prob_2": round(float(prob[order[1]]), 3)})

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["path", "view", "confidence", "is_cardiac", "view_2", "prob_2"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")
    ncard = sum(1 for r in rows if r["is_cardiac"] is True)
    print(f"gate: {ncard}/{len(rows)} classified cardiac; {len(rows)-ncard} non-cardiac/indeterminate")

    if args.labels:
        truth = {os.path.basename(r["filename"]): r["view"] for r in csv.DictReader(open(args.labels))}
        pairs = [(truth[os.path.basename(r["path"])], r["view"]) for r in rows
                 if os.path.basename(r["path"]) in truth and r["view"] != "READ_ERROR"]
        if pairs:
            acc = sum(a == b for a, b in pairs) / len(pairs)
            print(f"\nvs your labels (n={len(pairs)}): accuracy = {acc:.3f}")
            cm = {c: {d: 0 for d in CLASSES} for c in CLASSES}
            for a, b in pairs:
                if a in cm and b in cm[a]: cm[a][b] += 1
            print("confusion (rows=your label, cols=predicted):")
            print("        " + " ".join(f"{c[:4]:>5s}" for c in CLASSES))
            for c in CLASSES:
                print(f"{c[:6]:>6s}  " + " ".join(f"{cm[c][d]:5d}" for d in CLASSES))


if __name__ == "__main__":
    main()
