#!/usr/bin/env python3
"""
ED echo view + orientation + non-cardiac gate classifier — inference.

For each ultrasound loop (mp4) it outputs:
  view         : PSLA, PSSA, A4C, SX, IVC, Lung, Indeterminate   (orientation-invariant)
  confidence   : softmax probability of the chosen class
  is_cardiac   : gate — True for PSLA/PSSA/A4C/SX/IVC, False for Lung/Indeterminate
  orientation  : for PSLA/IVC only — 'EM' (emergency-medicine) or 'cards' (cardiology); else ''
  action       : gate_out | flip_then_use | use     (see below)

Actions
  - gate_out        : non-cardiac / uncertain -> exclude from downstream cardiac interpretation
  - flip_then_use   : a cardiac PSLA/IVC acquired in EM orientation -> horizontally flip to the
                      cardiology convention before feeding a cardiology-trained model (PanEcho/EchoPrime)
  - use             : cardiac view already in usable orientation

Usage
  python predict.py --input path/to/clips                 # folder of .mp4 (recurses)
  python predict.py --input clip.mp4                       # single loop
  python predict.py --input clips/ --labels labels.csv     # + agreement vs your view labels
                                                           # labels.csv columns: filename,view
"""
import argparse, csv, glob, os, sys
import numpy as np
import cv2
import torch
import torchvision

# 9 output classes (view + orientation for PSLA/IVC)
CLASSES = ["PSLA_EM", "PSLA_cards", "IVC_EM", "IVC_cards", "PSSA", "A4C", "SX", "Lung", "Indeterminate"]
CARDIAC = {"PSLA", "PSSA", "A4C", "SX", "IVC"}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def parse_class(name):
    """9-class name -> (view, orientation)."""
    if name.startswith(("PSLA", "IVC")):
        v, o = name.split("_")
        return v, o          # e.g. ('PSLA','EM')
    return name, ""          # congruent view, no orientation


def action_for(view, orientation, is_cardiac):
    if not is_cardiac:
        return "gate_out"
    if view in ("PSLA", "IVC") and orientation == "EM":
        return "flip_then_use"
    return "use"


def crop_and_scale(img, res=(224, 224), interpolation=cv2.INTER_CUBIC, zoom=0.1):
    in_res = (img.shape[1], img.shape[0]); r_in = in_res[0] / in_res[1]; r_out = res[0] / res[1]
    if r_in > r_out:
        pad = int(round((in_res[0] - r_out * in_res[1]) / 2)); img = img[:, pad:-pad]
    if r_in < r_out:
        pad = int(round((in_res[1] - in_res[0] / r_out) / 2)); img = img[pad:-pad]
    if zoom:
        px = int(img.shape[1] * zoom); py = int(img.shape[0] * zoom); img = img[py:-py, px:-px]
    return cv2.resize(img, res, interpolation=interpolation)

# NOTE on preprocessing: this model was trained on the center-crop above applied to QPath-style
# ~800x600 exports. Applying a different crop at inference (e.g. ultrasound-sector detection)
# shifts inputs off the training distribution and DEGRADES accuracy — verified empirically. Robust
# multi-format support requires a *sector-consistent retrain* (apply sector detection at both train
# and inference), not an inference-only patch. Until then, feed frames in the trained format.


def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    return torch.device("cpu")


def load_model(weights, device):
    ck = torch.load(weights, map_location="cpu")
    m = torchvision.models.convnext_tiny()
    m.classifier[-1] = torch.nn.Linear(m.classifier[-1].in_features, len(CLASSES))
    m.load_state_dict({k: v.float() for k, v in ck["model"].items()})
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
    ap.add_argument("--weights", default=os.path.join(os.path.dirname(__file__), "weights", "ed_view_orient_fp16.pt"))
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--labels", default=None, help="optional CSV with columns: filename,view (7-view vocab)")
    args = ap.parse_args()

    device = get_device(); print(f"device: {device}")
    model = load_model(args.weights, device)
    clips = find_clips(args.input)
    print(f"clips found: {len(clips)}")
    if not clips: sys.exit("no .mp4 clips found under --input")

    FIELDS = ["path", "view", "confidence", "is_cardiac", "orientation", "action", "view_2", "prob_2"]
    rows = []
    with torch.no_grad():
        for p in clips:
            fr = first_frame(p)
            if fr is None:
                rows.append({"path": p, "view": "READ_ERROR", "confidence": 0, "is_cardiac": "",
                             "orientation": "", "action": "", "view_2": "", "prob_2": ""}); continue
            x = torch.from_numpy(crop_and_scale(fr)).permute(2, 0, 1).float() / 255.0
            x = ((x - MEAN) / STD).unsqueeze(0).to(device)
            prob = torch.softmax(model(x), dim=1)[0].cpu().numpy()
            order = np.argsort(-prob)
            view, orient = parse_class(CLASSES[order[0]])
            is_card = view in CARDIAC
            view2, _ = parse_class(CLASSES[order[1]])
            rows.append({"path": p, "view": view, "confidence": round(float(prob[order[0]]), 3),
                         "is_cardiac": is_card, "orientation": orient,
                         "action": action_for(view, orient, is_card),
                         "view_2": view2, "prob_2": round(float(prob[order[1]]), 3)})

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(f"wrote {args.out} ({len(rows)} rows)")

    ncard = sum(1 for r in rows if r["is_cardiac"] is True)
    nflip = sum(1 for r in rows if r["action"] == "flip_then_use")
    print(f"gate: {ncard}/{len(rows)} cardiac ({len(rows)-ncard} gated out); "
          f"{nflip} cardiac loop(s) flagged EM-orientation -> flip before cardiology-model use")

    if args.labels:
        truth = {os.path.basename(r["filename"]): r["view"] for r in csv.DictReader(open(args.labels))}
        pairs = [(truth[os.path.basename(r["path"])], r["view"]) for r in rows
                 if os.path.basename(r["path"]) in truth and r["view"] != "READ_ERROR"]
        if pairs:
            acc = sum(a == b for a, b in pairs) / len(pairs)
            print(f"\nvs your VIEW labels (n={len(pairs)}): accuracy = {acc:.3f}")
            VIEWS = ["PSLA", "PSSA", "A4C", "SX", "IVC", "Lung", "Indeterminate"]
            cm = {c: {d: 0 for d in VIEWS} for c in VIEWS}
            for a, b in pairs:
                if a in cm and b in cm[a]: cm[a][b] += 1
            print("confusion (rows=your label, cols=predicted):")
            print("        " + " ".join(f"{c[:4]:>5s}" for c in VIEWS))
            for c in VIEWS:
                print(f"{c[:6]:>6s}  " + " ".join(f"{cm[c][d]:5d}" for d in VIEWS))


if __name__ == "__main__":
    main()
