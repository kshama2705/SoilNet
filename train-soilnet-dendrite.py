#!/usr/bin/env python3
# SoilNet/train-soilnet-dendrite.py
#
# MobileNetV2 + PerforatedAI dendrites for WoodScape Soiling (4 classes).
# Adds: eval-only mode, safe full-object load on PyTorch>=2.6, state_dict fallback,
# Conv2d+Linear dendrites, capacity-test OFF, history-switch OFF, sys.exit guard.

import argparse, csv, os, sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models.mobilenetv2 import MobileNetV2

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# PerforatedAI
from perforatedai import globals_perforatedai as PBG
from perforatedai import utils_perforatedai   as PBU
from torch.optim.lr_scheduler import CosineAnnealingLR

# --------- CLI ---------
def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", help="Train CSV (required unless --eval_only)")
    ap.add_argument("--test_csv",  required=True, help="Test CSV (also used for val-only metric if eval_only)")
    ap.add_argument("--out_dir",   default="runs/mobilenetv2_dendrite")
    ap.add_argument("--epochs",    type=int, default=20)
    ap.add_argument("--batch_size",type=int, default=64)
    ap.add_argument("--lr",        type=float, default=1e-4)
    ap.add_argument("--wd",        type=float, default=1e-4)
    ap.add_argument("--img_size",  type=int, default=224)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed",      type=int, default=42)

    # New flags
    ap.add_argument("--eval_only", action="store_true",
                    help="Skip training; only evaluate test set using saved weights from out_dir or --weights")
    ap.add_argument("--weights", type=str, default="",
                    help="Optional explicit path to weights (best_model_full.pt). If not set, will use out_dir defaults.")
    return ap.parse_args()

# --------- Dataset ---------
class SoilingCSV(Dataset):
    def __init__(self, csv_path: str, class_to_idx: dict, transform=None):
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.samples: List[Tuple[str, int]] = []
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            _ = next(reader)  # header
            for row in reader:
                if not row: continue
                img_path = row[0].strip()
                label_name = row[1].strip()
                if label_name in class_to_idx and os.path.isfile(img_path):
                    self.samples.append((img_path, class_to_idx[label_name]))
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {csv_path}")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx: int):
        p, y = self.samples[idx]
        img = Image.open(p).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, y

# --------- Utils ---------
def build_transforms(img_size=224):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return train_tf, eval_tf

def epoch_metrics(y_true, y_pred, average="macro"):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec  = recall_score(  y_true, y_pred, average=average, zero_division=0)
    f1   = f1_score(      y_true, y_pred, average=average, zero_division=0)
    return acc, prec, rec, f1

def plot_curve(xs, ys, title, ylabel, out_path):
    plt.figure()
    plt.plot(xs, ys, marker="o")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

# --------- PAI Config ---------
def pai_set_safety_and_prefs(model, num_classes=4):
    """Safe defaults: Conv2d+Linear wrapped, prompts off, capacity test off, history switch off."""
    # silence weight-decay prompt
    if hasattr(PBG, "pc") and hasattr(PBG.pc, "set_weight_decay_accepted"):
        PBG.pc.set_weight_decay_accepted(True)

    # allow dendrites for Conv2d and Linear
    if hasattr(PBG, "pc") and hasattr(PBG.pc, "set_module_types_to_convert"):
        PBG.pc.set_module_types_to_convert([nn.Linear, nn.Conv2d])
        if hasattr(PBG.pc, "set_module_types_to_track"):
            PBG.pc.set_module_types_to_track([nn.Linear, nn.Conv2d])

    if hasattr(PBG, "pc") and hasattr(PBG.pc, "set_module_names_to_convert"):
        PBG.pc.set_module_names_to_convert(["Linear", "Conv2d"])
        if hasattr(PBG.pc, "set_module_names_to_track"):
            PBG.pc.set_module_names_to_track(["Linear", "Conv2d"])

    # fallback: whitelist all Conv2d by name
    conv2d_names = [name for name, m in model.named_modules() if isinstance(m, nn.Conv2d)]
    if hasattr(PBG, "pc") and hasattr(PBG.pc, "set_specific_modules_to_convert"):
        PBG.pc.set_specific_modules_to_convert(conv2d_names)

    # skip interactive prompt and capacity test
    if hasattr(PBG, "pc") and hasattr(PBG.pc, "set_unwrapped_modules_confirmed"):
        PBG.pc.set_unwrapped_modules_confirmed(True)
    if hasattr(PBG, "pc") and hasattr(PBG.pc, "set_testing_dendrite_capacity"):
        PBG.pc.set_testing_dendrite_capacity(False)

    # disable history-based switching/import
    if hasattr(PBG, "pai_tracker"):
        if hasattr(PBG.pai_tracker, "set_import_best_on_switch"):
            PBG.pai_tracker.set_import_best_on_switch(False)
        if hasattr(PBG.pai_tracker, "set_use_history"):
            PBG.pai_tracker.set_use_history(False)
    if hasattr(PBG, "pc") and hasattr(PBG.pc, "set_switch_mode"):
        for mode in ("DOING_NOTHING", "DOING_SWITCH_EVERY_TIME", "n"):  # safest first
            try:
                PBG.pc.set_switch_mode(mode)
                break
            except Exception:
                pass

    # Disable breakpoints from library
    os.environ.setdefault("PYTHONBREAKPOINT", "0")

def initialize_pai_model(model):
    pai_set_safety_and_prefs(model)
    model = PBU.initialize_pai(model)
    return model

def save_best(model, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    full_path = out_dir / "best_model_full.pt"
    sd_path   = out_dir / "best_model_state_dict.pth"
    torch.save(model, full_path)
    torch.save(model.state_dict(), sd_path)

def load_best(device, out_dir: Path, num_classes=4, explicit_weights_path: str = ""):
    """Try full-object load (safe-globals), fallback to state_dict with re-wrap."""
    full_path = Path(explicit_weights_path) if explicit_weights_path else (out_dir / "best_model_full.pt")
    sd_path   = out_dir / "best_model_state_dict.pth"

    # Allow-list known classes for PyTorch>=2.6 weights_only behavior
    try:
        from torch.serialization import add_safe_globals
        add_safe_globals([MobileNetV2])
    except Exception:
        pass  # older torch, or API not present

    # Attempt full-object load
    if full_path.exists():
        try:
            model = torch.load(full_path, map_location=device, weights_only=False)
            print(f"[LOAD] Loaded full model object from {full_path}")
            return model.to(device).eval()
        except Exception as e:
            print(f"[LOAD] Full-object load failed ({e}). Will try state_dict fallback...")

    # Fallback: rebuild + wrap + load state_dict
    if not sd_path.exists():
        raise FileNotFoundError(f"No checkpoint found. Expected {full_path} or {sd_path}")
    print(f"[LOAD] Falling back to state_dict from {sd_path}")

    # Rebuild base arch
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    # IMPORTANT: wrap before load so tensors align with wrapped graph
    model = initialize_pai_model(model).to(device).eval()

    state = torch.load(sd_path, map_location=device, weights_only=True)
    # some PAI layers may introduce buffers; be lenient
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[LOAD] state_dict loaded with missing={len(missing)} unexpected={len(unexpected)}")
    return model

# --------- Main ---------
def main():
    args = get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Classes
    classes = ["clean", "transparent", "semi_transparent", "opaque"]
    class_to_idx = {c:i for i,c in enumerate(classes)}
    num_classes = len(classes)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms
    tf_train, tf_eval = build_transforms(args.img_size)

    # Datasets / loaders (build only what we need)
    ds_test = SoilingCSV(args.test_csv, class_to_idx, transform=tf_eval)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    # ----------------- Eval-only path -----------------
    if args.eval_only:
        model = load_best(device, out_dir, num_classes, explicit_weights_path=args.weights)
        # Evaluate
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in dl_test:
                xb = xb.to(device)
                logits = model(xb)
                y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
                y_true.extend(yb.numpy())
        test_acc, test_prec, test_rec, test_f1 = epoch_metrics(y_true, y_pred)
        with open(out_dir / "test_metrics.txt", "w") as f:
            f.write(f"Test Accuracy:  {test_acc:.4f}\n")
            f.write(f"Test Precision: {test_prec:.4f}\n")
            f.write(f"Test Recall:    {test_rec:.4f}\n")
            f.write(f"Test F1:        {test_f1:.4f}\n")
        print(f"[TEST] Acc {test_acc:.3f} | Prec {test_prec:.3f} | Rec {test_rec:.3f} | F1 {test_f1:.3f}")
        print(f"Artifacts saved to: {out_dir}")
        return

    # ----------------- Training path -----------------
    if not args.train_csv:
        raise ValueError("--train_csv is required when not using --eval_only")

    # Build train/val split
    ds_full_tmp = SoilingCSV(args.train_csv, class_to_idx, transform=tf_eval)
    labels = np.array([y for _, y in ds_full_tmp.samples]); idxs = np.arange(len(ds_full_tmp))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=args.seed)
    idx_train, idx_val = next(sss.split(idxs, labels))
    train_samples = [ds_full_tmp.samples[i] for i in idx_train]
    val_samples   = [ds_full_tmp.samples[i] for i in idx_val]

    class _Wrap(Dataset):
        def __init__(self, samples, transform): self.samples, self.transform = samples, transform
        def __len__(self): return len(self.samples)
        def __getitem__(self, i):
            p,y = self.samples[i]
            img = Image.open(p).convert("RGB")
            return self.transform(img), y

    ds_train = _Wrap(train_samples, tf_train)
    ds_val   = _Wrap(val_samples,   tf_eval)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    # Model
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model = model.to(device)

    # PAI: set prefs & wrap
    model = initialize_pai_model(model).to(device)

    # Optimizer/scheduler via tracker
    if hasattr(PBG, "pai_tracker"):
        PBG.pai_tracker.set_optimizer(torch.optim.AdamW)
        PBG.pai_tracker.set_scheduler(CosineAnnealingLR)
    optimArgs = {'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.wd}
    schedArgs = {'T_max': args.epochs}
    optimizer, scheduler = PBG.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
    criterion = nn.CrossEntropyLoss()

    # Logging
    log_path = out_dir / "metrics_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch",
            "train_loss","train_acc","train_prec","train_rec","train_f1",
            "val_loss","val_acc","val_prec","val_rec","val_f1"
        ])

    best_val_f1 = -1.0
    tr_losses, vl_losses, tr_accs, vl_accs, tr_f1s, vl_f1s = [], [], [], [], [], []

    # LAST-resort guard: intercept sys.exit from PAI internals
    _real_sys_exit = sys.exit
    def _pai_guarded_exit(code=0):
        print(f"[PAI] intercepted sys.exit({code}); continuing training")
        raise RuntimeError(f"PAI attempted sys.exit({code})")
    sys.exit = _pai_guarded_exit

    # Train
    for epoch in range(1, args.epochs+1):
        model.train()
        running_loss = 0.0
        y_true_tr, y_pred_tr = [], []

        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            y_true_tr.extend(yb.cpu().numpy())
            y_pred_tr.extend(torch.argmax(logits, dim=1).cpu().numpy())

        train_loss = running_loss / len(ds_train)
        train_acc, train_prec, train_rec, train_f1 = epoch_metrics(y_true_tr, y_pred_tr)

        # Val
        model.eval()
        running_val_loss = 0.0
        y_true_v, y_pred_v = [], []
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                running_val_loss += loss.item() * xb.size(0)
                y_true_v.extend(yb.cpu().numpy())
                y_pred_v.extend(torch.argmax(logits, dim=1).cpu().numpy())

        val_loss = running_val_loss / len(ds_val)
        val_acc, val_prec, val_rec, val_f1 = epoch_metrics(y_true_v, y_pred_v)

        # PAI tracker decision (guarded)
        val_score_pct = 100.0 * val_f1
        try:
            model, restructured, trainingComplete = PBG.pai_tracker.add_validation_score(val_score_pct, model)
        except (SystemExit, RuntimeError) as e:
            print(f"[PAI] Caught exit from tracker: {e}. Disabling history/switch and continuing.")
            if hasattr(PBG, "pai_tracker"):
                if hasattr(PBG.pai_tracker, "set_import_best_on_switch"):
                    PBG.pai_tracker.set_import_best_on_switch(False)
                if hasattr(PBG.pai_tracker, "set_use_history"):
                    PBG.pai_tracker.set_use_history(False)
            if hasattr(PBG, "pc") and hasattr(PBG.pc, "set_switch_mode"):
                try: PBG.pc.set_switch_mode("DOING_NOTHING")
                except Exception: pass
            restructured, trainingComplete = False, False

        model = model.to(device)
        if restructured:
            optimizer, scheduler = PBG.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)
            print(f"[PAI] Model restructured at epoch {epoch}. Rebuilt optimizer/scheduler.")

        # Log
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch,
                f"{train_loss:.6f}", f"{train_acc:.6f}", f"{train_prec:.6f}", f"{train_rec:.6f}", f"{train_f1:.6f}",
                f"{val_loss:.6f}",   f"{val_acc:.6f}",   f"{val_prec:.6f}",   f"{val_rec:.6f}",   f"{val_f1:.6f}"
            ])

        tr_losses.append(train_loss); vl_losses.append(val_loss)
        tr_accs.append(train_acc);   vl_accs.append(val_acc)
        tr_f1s.append(train_f1);     vl_f1s.append(val_f1)

        print(f"Epoch {epoch:02d}/{args.epochs} | "
              f"Train L {train_loss:.4f} A {train_acc:.3f} F1 {train_f1:.3f}  |  "
              f"Val L {val_loss:.4f} A {val_acc:.3f} F1 {val_f1:.3f}")

        # Save best (full object + state_dict)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_best(model, out_dir)

        if scheduler is not None:
            scheduler.step()

    # Curves
    epochs_axis = list(range(1, len(tr_losses)+1))
    plot_curve(epochs_axis, tr_losses, "Train Loss", "Loss", out_dir / "train_loss.png")
    plot_curve(epochs_axis, vl_losses, "Val Loss",   "Loss", out_dir / "val_loss.png")
    plot_curve(epochs_axis, tr_accs,  "Accuracy (Train)", "Accuracy", out_dir / "train_acc.png")
    plot_curve(epochs_axis, vl_accs,  "Accuracy (Val)",   "Accuracy", out_dir / "val_acc.png")
    plot_curve(epochs_axis, tr_f1s,   "F1 (Train)", "F1", out_dir / "train_f1.png")
    plot_curve(epochs_axis, vl_f1s,   "F1 (Val)",   "F1", out_dir / "val_f1.png")

    # Final Test (load best then eval)
    model = load_best(device, out_dir, num_classes)
    model = model.to(device).eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in dl_test:
            xb = xb.to(device)
            logits = model(xb)
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
            y_true.extend(yb.numpy())

    test_acc, test_prec, test_rec, test_f1 = epoch_metrics(y_true, y_pred)
    with open(out_dir / "test_metrics.txt", "w") as f:
        f.write(f"Test Accuracy:  {test_acc:.4f}\n")
        f.write(f"Test Precision: {test_prec:.4f}\n")
        f.write(f"Test Recall:    {test_rec:.4f}\n")
        f.write(f"Test F1:        {test_f1:.4f}\n")
    print(f"[TEST] Acc {test_acc:.3f} | Prec {test_prec:.3f} | Rec {test_rec:.3f} | F1 {test_f1:.3f}")
    print(f"Artifacts saved to: {out_dir}")

if __name__ == "__main__":
    main()
