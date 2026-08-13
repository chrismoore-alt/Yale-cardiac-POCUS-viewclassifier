#!/usr/bin/env python3
# Build the view+orientation dataset from the existing 7-class frame cache.
# Selective flip: PSLA and IVC get an EM (original) and a cardiology (h-flipped) variant;
# congruent views (PSSA/A4C/SX/Lung/Indeterminate) are used as-is (NOT flipped).
# 9 classes -> view is recoverable, and PSLA/IVC carry an orientation.
import numpy as np, os

SRC = "/Users/christophermoore/ai_echo_local/view_ds/view_dataset.npz"
OUT = "/Users/christophermoore/ai_echo_local/view_ds/orient_dataset.npz"

d = np.load(SRC, allow_pickle=True)
X, y, cases, split = d["X"], d["y"], d["cases"], d["split"]
C7 = [str(c) for c in d["classes"]]                      # PSLA,PSSA,A4C,SX,IVC,Lung,Indeterminate
C9 = ["PSLA_EM", "PSLA_cards", "IVC_EM", "IVC_cards", "PSSA", "A4C", "SX", "Lung", "Indeterminate"]
I9 = {c: i for i, c in enumerate(C9)}

Xo, yo, co, so = [], [], [], []
for i in range(len(X)):
    v = C7[y[i]]
    if v in ("PSLA", "IVC"):
        Xo.append(X[i]);              yo.append(I9[f"{v}_EM"]);    co.append(cases[i]); so.append(split[i])
        Xo.append(np.fliplr(X[i]));   yo.append(I9[f"{v}_cards"]); co.append(cases[i]); so.append(split[i])
    else:
        Xo.append(X[i]);              yo.append(I9[v]);            co.append(cases[i]); so.append(split[i])

Xo = np.stack(Xo); yo = np.array(yo, np.int64); co = np.array(co); so = np.array(so)
print(f"orientation dataset: X={Xo.shape}")
print("class counts:", {C9[i]: int((yo == i).sum()) for i in range(len(C9))})
for s in ["train", "val", "test"]:
    m = so == s
    print(f"  {s}: {int(m.sum())} frames, {len(set(co[m]))} cases")
np.savez_compressed(OUT, X=Xo, y=yo, cases=co, split=so, classes=np.array(C9))
print(f"saved -> {OUT} ({os.path.getsize(OUT)/1e6:.0f} MB)")
