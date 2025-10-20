#!/usr/bin/env python3
# train_mobilenetv2_woodscape.py
#
# Train MobileNetV2 on WoodScape Soiling 4-class labels produced by the CSV generator.
# - Reads train CSV (image_path,label,clean_pct,transparent_pct,semi_transparent_pct,opaque_pct)
# - Splits train -> train/val
# - Evaluates on test CSV
# - Saves curves (loss, acc, f1) and a metrics log CSV
#
# Example:
#   python train_mobilenetv2_woodscape.py \
#       --train_csv soiling_dataset/soiling_labels_4class_train.csv \
#       --test_csv  soiling_dataset/soiling_labels_4class_test.csv \
#       --out_dir   runs/mobilenetv2_soiling \
#       --epochs 20 --batch_size 64

import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
try:
    from sklearn.model_selection import StratifiedShuffleSplit
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------
# Dataset from CSV
# ---------------------------
class SoilingCSV(Dataset):
    def __init__(self, csv_path: str, class_to_idx: dict, transform=None):
        self.csv_path = csv_path
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.samples: List[Tuple[str, int]] = []

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)  # image_path,label,...
            for row in reader:
                if not row: continue
                img_path = row[0].strip()
                label_name = row[1].strip()
                if label_name not in class_to_idx:
                    # unseen class? skip
                    continue
                self.samples.append((img_path, class_to_idx[label_name]))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {csv_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, y = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform: img = self.transform(img)
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
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size*1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return train_tf, eval_tf


def split_train_val(dataset: SoilingCSV, val_ratio: float, seed: int=42):
    # Prefer stratified split (keeps class balance)
    if HAS_SKLEARN:
        labels = np.array([y for _, y in dataset.samples])
        idxs = np.arange(len(dataset))
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
        train_idx, val_idx = next(sss.split(idxs, labels))
        return Subset(dataset, train_idx), Subset(dataset, val_idx)
    else:
        # Fallback to torch's random split (not stratified)
        n_total = len(dataset)
        n_val = int(n_total * val_ratio)
        n_train = n_total - n_val
        gen = torch.Generator().manual_seed(seed)
        return torch.utils.data.random_split(dataset, [n_train, n_val], generator=gen)


def epoch_metrics(y_true, y_pred, average="macro"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average, zero_division=0)
    rec = recall_score(y_true, y_pred, average=average, zero_division=0)
    f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
    return acc, prec, rec, f1


def evaluate(model, loader, device):
    model.eval()
    ys, yh = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            ys.extend(y.numpy())
            yh.extend(pred)
    return epoch_metrics(ys, yh)


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
    ap.add_argument("--train_csv", required=True, help="CSV with train image paths & labels")
    ap.add_argument("--test_csv", required=True, help="CSV with test image paths & labels")
    ap.add_argument("--out_dir",  default="runs/mobilenetv2_soiling")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--val_ratio", type=float, default=0.15, help="fraction taken from train for validation")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # fixed class order to match your CSV labels
    classes = ["clean", "transparent", "semi_transparent", "opaque"]
    class_to_idx = {c:i for i,c in enumerate(classes)}
    idx_to_class = {i:c for c,i in class_to_idx.items()}
    num_classes = len(classes)

    # transforms
    tf_train, tf_eval = build_transforms(args.img_size)

    # datasets
    ds_train_full = SoilingCSV(args.train_csv, class_to_idx, transform=tf_eval)  # temp eval tf
    # Make split on indices (using current transform), then wrap subsets with their own transforms
    train_subset, val_subset = split_train_val(ds_train_full, args.val_ratio, seed=args.seed)
    # Rewrap to apply different transforms
    train_subset = SoilingCSV(args.train_csv, class_to_idx, transform=tf_train)
    # Keep only train indices in the wrapped dataset
    if isinstance(train_subset, SoilingCSV) and HAS_SKLEARN:
        # Rebuild samples using selected indices from stratified split
        # Load indices again
        labels = np.array([y for _, y in ds_train_full.samples])
        idxs = np.arange(len(ds_train_full))
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=args.seed)
        idx_train, idx_val = next(sss.split(idxs, labels))
        # Filter samples
        train_samples = [ds_train_full.samples[i] for i in idx_train]
        val_samples   = [ds_train_full.samples[i] for i in idx_val]
        # Wrap two tiny datasets with proper transforms
        class _Wrap(Dataset):
            def __init__(self, samples, transform): self.samples, self.transform = samples, transform
            def __len__(self): return len(self.samples)
            def __getitem__(self, i):
                p,y = self.samples[i]
                img = Image.open(p).convert("RGB")
                return self.transform(img), y
        ds_train = _Wrap(train_samples, tf_train)
        ds_val   = _Wrap(val_samples,   tf_eval)
    else:
        # torch random_split fallback
        n_total = len(ds_train_full)
        n_val = int(n_total * args.val_ratio)
        n_train = n_total - n_val
        gen = torch.Generator().manual_seed(args.seed)
        train_idx, val_idx = torch.utils.data.random_split(range(n_total), [n_train, n_val], generator=gen)
        class _Wrap(Dataset):
            def __init__(self, base, indices, transform): self.base, self.idx, self.transform = base, indices, transform
            def __len__(self): return len(self.idx)
            def __getitem__(self, i):
                p,y = self.base.samples[self.idx[i]]
                img = Image.open(p).convert("RGB")
                return self.transform(img), y
        ds_train = _Wrap(ds_train_full, list(train_idx), tf_train)
        ds_val   = _Wrap(ds_train_full, list(val_idx),   tf_eval)

    ds_test = SoilingCSV(args.test_csv, class_to_idx, transform=tf_eval)

    # loaders
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    # model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # logs
    log_path = Path(args.out_dir) / "metrics_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss","train_acc","train_prec","train_rec","train_f1",
            "val_loss","val_acc","val_prec","val_rec","val_f1"
        ])

    best_val_f1 = -1.0
    tr_losses, vl_losses, tr_accs, vl_accs, tr_f1s, vl_f1s = [], [], [], [], [], []

    # training
    for epoch in range(1, args.epochs+1):
        model.train()
        running_loss = 0.0
        y_true_tr, y_pred_tr = [], []

        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            y_true_tr.extend(yb.cpu().numpy())
            y_pred_tr.extend(preds.cpu().numpy())

        train_loss = running_loss / len(ds_train)
        train_acc, train_prec, train_rec, train_f1 = epoch_metrics(y_true_tr, y_pred_tr)

        # val pass
        model.eval()
        running_val_loss = 0.0
        y_true_v, y_pred_v = [], []
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                running_val_loss += loss.item() * xb.size(0)
                preds = torch.argmax(logits, dim=1)
                y_true_v.extend(yb.cpu().numpy())
                y_pred_v.extend(preds.cpu().numpy())

        val_loss = running_val_loss / len(ds_val)
        val_acc, val_prec, val_rec, val_f1 = epoch_metrics(y_true_v, y_pred_v)

        # log
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
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

        # save best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), str(Path(args.out_dir) / "best_mobilenetv2.pth"))

    # save curves
    epochs = list(range(1, args.epochs+1))
    plot_curve(epochs, tr_losses, "Train Loss", "Loss", Path(args.out_dir)/"train_loss.png")
    plot_curve(epochs, vl_losses, "Val Loss", "Loss", Path(args.out_dir)/"val_loss.png")
    plot_curve(epochs, tr_accs,  "Accuracy (Train)", "Accuracy", Path(args.out_dir)/"train_acc.png")
    plot_curve(epochs, vl_accs,  "Accuracy (Val)",   "Accuracy", Path(args.out_dir)/"val_acc.png")
    plot_curve(epochs, tr_f1s,   "F1 (Train)", "F1", Path(args.out_dir)/"train_f1.png")
    plot_curve(epochs, vl_f1s,   "F1 (Val)",   "F1", Path(args.out_dir)/"val_f1.png")

    # final test evaluation with best weights
    best_path = Path(args.out_dir) / "best_mobilenetv2.pth"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
    test_acc, test_prec, test_rec, test_f1 = evaluate(model, dl_test, device)
    with open(Path(args.out_dir) / "test_metrics.txt", "w") as f:
        f.write(f"Test Accuracy:  {test_acc:.4f}\n")
        f.write(f"Test Precision: {test_prec:.4f}\n")
        f.write(f"Test Recall:    {test_rec:.4f}\n")
        f.write(f"Test F1:        {test_f1:.4f}\n")
    print(f"[TEST] Acc {test_acc:.3f} | Prec {test_prec:.3f} | Rec {test_rec:.3f} | F1 {test_f1:.3f}")
    print(f"Artifacts saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
