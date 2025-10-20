#!/usr/bin/env python3
import argparse, csv
from pathlib import Path
from typing import List, Tuple
from collections import Counter
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------- Dataset ----------
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
                if label_name in class_to_idx:
                    self.samples.append((img_path, class_to_idx[label_name]))
        if not self.samples:
            raise RuntimeError(f"No samples found in {csv_path}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx: int):
        p, y = self.samples[idx]
        img = Image.open(p).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, y


def build_eval_tf(img_size=224):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def plot_confusion(cm: np.ndarray, classes: list, out_path: Path, normalize: bool = True):
    if normalize:
        with np.errstate(all='ignore'):
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=30, ha='right')
    plt.yticks(tick_marks, classes)

    thresh = cm.max() * 0.6
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            plt.text(j, i, txt,
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=9)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--out_dir", default="runs/eval_mobilenetv2")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    # Controls averaging style so you can match training logs exactly
    ap.add_argument("--average", choices=["macro", "weighted", "micro"], default="macro",
                    help="Averaging mode for 'present-class' metrics.")
    ap.add_argument("--force_all_four", action="store_true",
                    help="Also report fixed-4-class metrics (missing classes count as 0).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    classes = ["clean", "transparent", "semi_transparent", "opaque"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    label_indices = list(range(len(classes)))  # [0,1,2,3]

    # Data
    tf_eval = build_eval_tf(args.img_size)
    ds_test = SoilingCSV(args.test_csv, class_to_idx, transform=tf_eval)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(classes))
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model = model.to(device).eval()

    # Inference
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in dl_test:
            xb = xb.to(device)
            logits = model(xb)
            y_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
            y_true.extend(yb.numpy())

    # Show test-set class distribution
    counts = Counter(y_true)
    with open(out_dir / "label_counts.txt", "w") as f:
        for i, c in enumerate(classes):
            f.write(f"{c:18s}: {counts.get(i, 0)}\n")

    # ---------------------------
    # 1) Present-class metrics (matches training behavior)
    # ---------------------------
    acc_present  = accuracy_score(y_true, y_pred)
    prec_present = precision_score(y_true, y_pred, average=args.average, zero_division=0)
    rec_present  = recall_score(  y_true, y_pred, average=args.average, zero_division=0)
    f1_present   = f1_score(      y_true, y_pred, average=args.average, zero_division=0)

    report_present = classification_report(
        y_true, y_pred,
        target_names=[classes[i] for i in sorted(set(y_true))],
        digits=4, zero_division=0
    )

    # ---------------------------
    # 2) Fixed-4-class metrics (optional)
    # ---------------------------
    metrics_fixed = {}
    report_fixed = None
    cm_raw = confusion_matrix(y_true, y_pred, labels=label_indices)
    plot_confusion(cm_raw, classes, out_dir / "confusion_matrix_norm.png", normalize=True)
    plot_confusion(cm_raw, classes, out_dir / "confusion_matrix_raw.png",  normalize=False)

    if args.force_all_four:
        acc_fixed  = accuracy_score(y_true, y_pred)
        prec_fixed = precision_score(y_true, y_pred, labels=label_indices,
                                     average="macro", zero_division=0)
        rec_fixed  = recall_score(  y_true, y_pred, labels=label_indices,
                                     average="macro", zero_division=0)
        f1_fixed   = f1_score(      y_true, y_pred, labels=label_indices,
                                     average="macro", zero_division=0)
        metrics_fixed = dict(acc=acc_fixed, prec=prec_fixed, rec=rec_fixed, f1=f1_fixed)

        report_fixed = classification_report(
            y_true, y_pred,
            labels=label_indices, target_names=classes,
            digits=4, zero_division=0
        )

    # Save everything
    with open(out_dir / "eval_metrics.txt", "w") as f:
        f.write("== Present-class metrics (matches training) ==\n")
        f.write(f"Accuracy : {acc_present:.4f}\n")
        f.write(f"Precision: {prec_present:.4f}  (average={args.average})\n")
        f.write(f"Recall   : {rec_present:.4f}   (average={args.average})\n")
        f.write(f"F1       : {f1_present:.4f}    (average={args.average})\n\n")
        f.write("Per-class report (present classes only):\n")
        f.write(report_present + "\n")

        f.write("\n== Fixed-4-class metrics (optional) ==\n")
        if args.force_all_four:
            f.write(f"Accuracy : {metrics_fixed['acc']:.4f}\n")
            f.write(f"Precision: {metrics_fixed['prec']:.4f}  (macro over 4 classes)\n")
            f.write(f"Recall   : {metrics_fixed['rec']:.4f}   (macro over 4 classes)\n")
            f.write(f"F1       : {metrics_fixed['f1']:.4f}    (macro over 4 classes)\n\n")
            f.write("Per-class report (all 4 classes; missing classes=0):\n")
            f.write(report_fixed + "\n")
        else:
            f.write("Skipped (run with --force_all_four to include).\n")

    print("=== Present-class (matches training) ===")
    print(f"Acc {acc_present:.4f} | Prec {prec_present:.4f} | Rec {rec_present:.4f} | F1 {f1_present:.4f}")
    if args.force_all_four:
        print("\n=== Fixed-4-class (missing classes penalized) ===")
        print(f"Acc {metrics_fixed['acc']:.4f} | Prec {metrics_fixed['prec']:.4f} | "
              f"Rec {metrics_fixed['rec']:.4f} | F1 {metrics_fixed['f1']:.4f}")
    print(f"\nSaved metrics & confusion matrices to: {out_dir}")


if __name__ == "__main__":
    main()
