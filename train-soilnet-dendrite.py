#!/usr/bin/env python3
# SoilNet/train-soilnet-dendrite.py
#
# MobileNetV2 + PerforatedAI dendrites for WoodScape Soiling (4 classes).
# Key change: **Eager dendrite conversion** — we directly call the PerforatedAI
# converter right after initialize_pai so dendrites are added immediately
# (no reliance on tracker "switch" heuristics).
#
# CSV format (from your converter):
#   image_path,label,clean_pct,transparent_pct,semi_transparent_pct,opaque_pct
#
# Example train:
#   python SoilNet/train-soilnet-dendrite.py \
#     --train_csv soiling_dataset/soiling_labels_4class_train.csv \
#     --test_csv  soiling_dataset/soiling_labels_4class_test.csv \
#     --out_dir   runs/mobilenetv2_soiling \
#     --epochs 60 --batch_size 64
#
# Example eval-only (re-uses saved best_model_full.pt and runs test):
#   python SoilNet/train-soilnet-dendrite.py \
#     --train_csv soiling_dataset/soiling_labels_4class_train.csv \
#     --test_csv  soiling_dataset/soiling_labels_4class_test.csv \
#     --out_dir   runs/mobilenetv2_soiling \
#     --eval_only 1

import argparse, csv, os, sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==== PerforatedAI (matches your installed modules) ====
from perforatedai import globals_perforatedai as PBG
from perforatedai import utils_perforatedai   as PBU
from torch.optim.lr_scheduler import CosineAnnealingLR


# ---------------------------
# Dataset from CSV
# ---------------------------
class SoilingCSV(Dataset):
    def __init__(self, csv_path: str, class_to_idx: dict, transform=None):
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.samples: List[Tuple[str, int]] = []

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            _ = next(reader)  # header
            for row in reader:
                if not row:
                    continue
                img_path = row[0].strip()
                label_name = row[1].strip()
                if label_name in class_to_idx and os.path.isfile(img_path):
                    self.samples.append((img_path, class_to_idx[label_name]))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        p, y = self.samples[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, y


# ---------------------------
# Utilities
# ---------------------------
def build_transforms(img_size=224):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def epoch_metrics(y_true, y_pred, average="macro"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
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


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--out_dir", default="runs/mobilenetv2_soiling")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_only", type=int, default=0, help="1 = skip training, evaluate saved best model")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- classes (fixed order) ----
    classes = ["clean", "transparent", "semi_transparent", "opaque"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)

    # ---- transforms ----
    tf_train, tf_eval = build_transforms(args.img_size)

    # ---- load full train set (to split stratified) ----
    ds_full_tmp = SoilingCSV(args.train_csv, class_to_idx, transform=tf_eval)
    labels = np.array([y for _, y in ds_full_tmp.samples])
    idxs = np.arange(len(ds_full_tmp))

    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=args.seed)
    idx_train, idx_val = next(sss.split(idxs, labels))
    train_samples = [ds_full_tmp.samples[i] for i in idx_train]
    val_samples = [ds_full_tmp.samples[i] for i in idx_val]

    class _Wrap(Dataset):
        def __init__(self, samples, transform):
            self.samples, self.transform = samples, transform
        def __len__(self): return len(self.samples)
        def __getitem__(self, i):
            p, y = self.samples[i]
            img = Image.open(p).convert("RGB")
            return self.transform(img), y

    ds_train = _Wrap(train_samples, tf_train)
    ds_val   = _Wrap(val_samples,   tf_eval)
    ds_test  = SoilingCSV(args.test_csv, class_to_idx, transform=tf_eval)

    # ---- loaders ----
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    # ---- model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model = model.to(device)

    # ==== PerforatedAI: allow weight decay & add dendrites eagerly ====
    GPA.pc.set_weight_decay_accepted(True)
    model = UPA.initialize_pai(model)
    model = model.to(device)

    # ---- optimizer / scheduler (via PAI tracker, but we avoid pdb prompts)
    PBG.pai_tracker.set_optimizer(torch.optim.AdamW)
    PBG.pai_tracker.set_scheduler(CosineAnnealingLR)
    optimArgs = {'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.wd}
    schedArgs = {'T_max': args.epochs}
    optimizer, scheduler = PBG.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)

    criterion = nn.CrossEntropyLoss()

    # Paths
    log_path = out_dir / "metrics_log.csv"
    best_full_path = out_dir / "best_model_full.pt"  # save full object to preserve dendrites

    # EVAL-ONLY: load and run test, then exit
    if args.eval_only:
        if not best_full_path.exists():
            print(f"[EVAL] Missing {best_full_path}. Train first.")
            sys.exit(1)
        # NOTE: we intentionally use weights_only=False here to load the full wrapped object
        model = torch.load(best_full_path, map_location=device, weights_only=False)
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
        return

    # ---- logging ----
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "epoch",
            "train_loss", "train_acc", "train_prec", "train_rec", "train_f1",
            "val_loss", "val_acc", "val_prec", "val_rec", "val_f1"
        ])

    best_val_f1 = -1.0
    tr_losses, vl_losses, tr_accs, vl_accs, tr_f1s, vl_f1s = [], [], [], [], [], []

    # ----------------- training loop -----------------
    print("Running Dendrite Experiment (eager-wrapped).")
    # Dendrites loop until training complete is True
    epoch = -1
    while True:
        epoch += 1
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

        # ---- validation ----
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

        model, restructured, training_complete = GPA.pai_tracker.add_validation_score(100.0 * val_f1, 
        model) # .module if its a dataParallel
        model.to(device)
        if(training_complete):
            break
        elif(restructured):
            optimArgs = {'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.wd}
            schedArgs = {'T_max': args.epochs}
            optimizer, scheduler = PBG.pai_tracker.setup_optimizer(model, optimArgs, schedArgs)


        # ---- log & print ----
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

        # Save best by val F1 — FULL OBJECT (preserves dendrites)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model, best_full_path)

        # CosineAnnealingLR: step once per epoch
        if scheduler is not None:
            scheduler.step()

    # ---- curves ----
    epochs_axis = list(range(1, len(tr_losses) + 1))
    plot_curve(epochs_axis, tr_losses, "Train Loss", "Loss", out_dir / "train_loss.png")
    plot_curve(epochs_axis, vl_losses, "Val Loss", "Loss", out_dir / "val_loss.png")
    plot_curve(epochs_axis, tr_accs, "Accuracy (Train)", "Accuracy", out_dir / "train_acc.png")
    plot_curve(epochs_axis, vl_accs, "Accuracy (Val)", "Accuracy", out_dir / "val_acc.png")
    plot_curve(epochs_axis, tr_f1s, "F1 (Train)", "F1", out_dir / "train_f1.png")
    plot_curve(epochs_axis, vl_f1s, "F1 (Val)", "F1", out_dir / "val_f1.png")

    # Final model is loaded automatically when add_validation score returns training_complete
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
