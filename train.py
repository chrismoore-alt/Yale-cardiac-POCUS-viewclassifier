# NOTE: exact script used to train the released model. Paths/IO reflect the original
# environment and expect the (non-distributed) expert-labeled loops. Adapt paths to reuse.

#!/usr/bin/env python3
# Train an ED-native view + gate classifier (7 classes incl. Lung/Indeterminate) on the
# expert clip-view labels. Fine-tunes ImageNet ConvNeXt-Tiny. NO horizontal-flip augmentation
# (the model learns ED orientation natively -> no best-of-both flip needed at inference).
import numpy as np, time, json
import torch, torch.nn as nn, torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as T
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

DS = "/path/to/ai_echo_local/view_ds/view_dataset.npz"
CKPT = "/path/to/ai_echo_local/view_ds/ed_view_convnext.pt"
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

d = np.load(DS, allow_pickle=True)
X, y, split = d["X"], d["y"], d["split"]
CLASSES = [str(c) for c in d["classes"]]; NC = len(CLASSES)
mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1); std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)

aug = T.Compose([T.RandomResizedCrop(224, scale=(0.80,1.0), ratio=(0.9,1.1), antialias=True),
                 T.ColorJitter(brightness=0.2, contrast=0.2)])

class VDS(Dataset):
    def __init__(self, mask, train):
        self.idx = np.where(mask)[0]; self.train = train
    def __len__(self): return len(self.idx)
    def __getitem__(self, i):
        j = self.idx[i]
        t = torch.from_numpy(X[j]).permute(2,0,1).float()/255.0   # CHW [0,1]
        if self.train: t = aug(t)
        return (t - mean)/std, int(y[j])

tr = DataLoader(VDS(split=="train", True),  batch_size=32, shuffle=True,  num_workers=0)
va = DataLoader(VDS(split=="val",  False),  batch_size=64, shuffle=False, num_workers=0)
te = DataLoader(VDS(split=="test", False),  batch_size=64, shuffle=False, num_workers=0)

model = torchvision.models.convnext_tiny(weights=torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NC)
model.to(dev)

# class-weighted loss
cnt = np.bincount(y[split=="train"], minlength=NC).astype(float)
w = torch.tensor((cnt.sum()/(NC*cnt)), dtype=torch.float, device=dev)
crit = nn.CrossEntropyLoss(weight=w)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
EPOCHS = 12
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

@torch.no_grad()
def evaluate(loader):
    model.eval(); P=[]; Yt=[]
    for xb,yb in loader:
        out = model(xb.to(dev)); P.append(out.argmax(1).cpu().numpy()); Yt.append(yb.numpy())
    P=np.concatenate(P); Yt=np.concatenate(Yt)
    return (P==Yt).mean(), cohen_kappa_score(Yt,P), P, Yt

best=0.0; t0=time.time()
for ep in range(1, EPOCHS+1):
    model.train(); tl=0; n=0
    for xb,yb in tr:
        opt.zero_grad()
        loss = crit(model(xb.to(dev)), yb.to(dev))
        loss.backward(); opt.step()
        tl += loss.item()*len(yb); n += len(yb)
    sched.step()
    acc,kap,_,_ = evaluate(va)
    print(f"ep{ep:2d} loss={tl/n:.3f} val_acc={acc:.3f} val_kappa={kap:.3f} ({(time.time()-t0)/60:.1f}min)", flush=True)
    if acc>best:
        best=acc; torch.save({"model":model.state_dict(),"classes":CLASSES}, CKPT)

# final test with best checkpoint
model.load_state_dict(torch.load(CKPT)["model"]);
acc,kap,P,Yt = evaluate(te)
print(f"\n=== TEST (n={len(Yt)}): acc={acc:.3f}  Cohen kappa={kap:.3f} ===")
print(classification_report(Yt, P, target_names=CLASSES, digits=3))
print("confusion (rows=true, cols=pred):"); print("      "+" ".join(f"{c[:4]:>5s}" for c in CLASSES))
cm = confusion_matrix(Yt, P)
for i,c in enumerate(CLASSES): print(f"{c[:5]:>5s} "+" ".join(f"{v:5d}" for v in cm[i]))
np.savez(f"/path/to/ai_echo_local/view_ds/test_preds.npz", pred=P, true=Yt, classes=np.array(CLASSES))
print(f"\nsaved model -> {CKPT}")
