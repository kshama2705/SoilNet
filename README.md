# SoilNet: MobileNetV2 + PerforatedAI Dendrites for Camera Soiling Classification

This project fine-tunes a **MobileNetV2** classifier on the **WoodScape Soiling dataset**, optionally enhanced with **PerforatedAI dendritic backpropagation** for improved efficiency and adaptive capacity.  
It classifies four lens soiling states:

- `clean`
- `transparent`
- `semi_transparent`
- `opaque`

---

## Dataset Setup

### 1. Download and Extract
Obtain the **WoodScape Soiling** subset (from [Valeo WoodScape](https://github.com/valeoai/WoodScape)) and extract it:
```
soiling_dataset/
  ├── train/
  │   ├── rgb_images/
  │   ├── soiling_masks/
  │   └── ...
  ├── test/
  │   ├── rgb_images/
  │   ├── soiling_masks/
  │   └── ...
  ├── soiling_annotation_info.json
```

### 2. Convert Annotations
Use the converter script to map masks → CSV labels:

```bash
python SoilNet/convert_masks_to_classes.py
```

This creates:
```
soiling_dataset/soiling_labels_4class_train.csv
soiling_dataset/soiling_labels_4class_test.csv
```

Each row looks like:
```
/home/ubuntu/soiling_dataset/train/rgb_images/0001_FV.png,clean,0.87,0.05,0.07,0.00
```

---

## Environment Setup

```bash
conda create -n SoilNet python=3.11
conda activate SoilNet
pip install torch torchvision scikit-learn matplotlib pillow pandas perforatedai
```

> If you have a PerforatedAI license file (`license.rsa` or similar), place it in your working directory.

Export your credentials if required:
```bash
export PAIPASSWORD="your_password"
# or
export PAIEMAIL="you@domain.com"
export PAITOKEN="xxxx-xxxx"
```

---

## Training

### Baseline (without dendrites)
```bash
python SoilNet/train-soilnet.py   --train_csv soiling_dataset/soiling_labels_4class_train.csv   --test_csv  soiling_dataset/soiling_labels_4class_test.csv   --out_dir   runs/mobilenetv2_baseline   --epochs 20 --batch_size 64 --lr 1e-4 --wd 1e-4   --img_size 224 --val_ratio 0.15 --num_workers 4 --seed 42
```

### Dendritic version (PerforatedAI)
```bash
python SoilNet/train-soilnet-dendrite.py --train_csv soiling_dataset/soiling_labels_4class_train.csv --test_csv soiling_dataset/soiling_labels_4class_test.csv --out_dir runs/mobilenetv2_soiling --epochs 60 --batch_size 64
```

If prompted with messages like:
```
Modules that are not wrapped will not have dendrites...
Type 'c' and hit Enter to continue
```
Type `c` and press **Enter** — this confirms the default safe wrapping.

---

## Dendritic Wrapping Details

By default, **only `nn.Linear` layers** are dendritically wrapped for stability.  
To include `Conv2d` layers as well, add this near the top of your script:

```python
from torch import nn
import perforatedai.globals_perforatedai as PBG
PBG.pc.set_module_types_to_convert([nn.Linear, nn.Conv2d])
PBG.pc.set_module_types_to_track([nn.Linear, nn.Conv2d])
```

---

## Outputs

All logs and checkpoints are saved under `runs/`.

| File | Description |
|------|--------------|
| `metrics_log.csv` | Per-epoch train/val loss and metrics |
| `best_mobilenetv2_dendrite.pth` | Best model (by val F1) |
| `test_metrics.txt` | Final accuracy, precision, recall, F1 |
| `train_loss.png` / `val_loss.png` | Loss curves |
| `train_acc.png` / `val_acc.png` | Accuracy curves |

---

## Evaluation

Once training completes:

```bash
python SoilNet/eval-soilnet.py   --weights runs/mobilenetv2_dendrite/best_mobilenetv2_dendrite.pth   --test_csv soiling_dataset/soiling_labels_4class_test.csv
```

This prints and saves:
```
Test Accuracy:  0.9432
Test Precision: 0.9410
Test Recall:    0.9455
Test F1:        0.9432
```

---

## Notes

- If you see this prompt:
  ```
  Building dendrites without Perforated Backpropagation
  ```
  It’s a harmless info message — means dendritic hooks are initializing.

- For stability, it’s best to start training on **a small number of epochs (5–10)** first, confirm loss decreases, then scale up.

---

## Citation

If you use this project or build upon it, please cite:

> Valeo WoodScape Dataset: “WoodScape: A Multi-Task, Multi-Camera Fisheye Dataset for Autonomous Driving.” ICCV 2019.

> PerforatedAI Dendritic Learning: PerforatedAI API Documentation – 2025, [https://github.com/PerforatedAI/PerforatedAI](https://github.com/PerforatedAI/PerforatedAI)

---

### Maintainer
**Kshama N. Shah**  
Magna International – AI / Robotics  
(For educational and research use)
