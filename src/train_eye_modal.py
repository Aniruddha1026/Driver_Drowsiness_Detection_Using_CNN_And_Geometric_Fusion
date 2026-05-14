"""
Week 3 — TinyEyeNet Training (Small Dataset Edition)
Project: Real-time Driver Drowsiness Detection

Changes vs original for small dataset (1680 train samples):
  - Stronger augmentation  → artificially expands effective dataset
  - LR       : 1e-3 → 5e-4  (smaller steps prevent overfitting)
  - Epochs   : 50   → 80    (small datasets need more epochs)
  - Patience : 10   → 15    (more patience before early stop)
  - Dropout  : 0.4  → 0.5   (stronger regularisation)
  - WeightDecay: 1e-4 → 1e-3 (stronger L2 penalty)
  - No class weights (dataset is 50/50 balanced)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (f1_score, precision_score,
                              recall_score, confusion_matrix)
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import json
import time
import random

# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

SPLITS_DIR   = pathlib.Path("data/splits/eye")
OUTPUT_DIR   = pathlib.Path("modals/eye3")

IMG_SIZE     = 24
BATCH_SIZE   = 16      # smaller batch — better gradient signal on small data
NUM_EPOCHS   = 80      # more epochs for small dataset
LR           = 5e-4    # lower LR — prevents overfitting
WEIGHT_DECAY = 1e-3    # stronger L2 regularisation
PATIENCE     = 15      # more patience before early stop
SEED         = 42

CLASS_NAMES  = ["closed", "open"]   # alphabetical: closed=0, open=1

# ═══════════════════════════════════════════════════════════════════
# REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════════

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

# ═══════════════════════════════════════════════════════════════════
# MODEL — TinyEyeNet
# ═══════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """Conv → BN → ReLU → MaxPool"""
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class TinyEyeNet(nn.Module):
    """
    1 × 24 × 24 → 2 classes (closed=0, open=1)

    Spatial dimensions:
      ConvBlock(1→8,   pool) →  8×12×12
      ConvBlock(8→16,  pool) → 16× 6× 6
      ConvBlock(16→32, pool) → 32× 3× 3
      GlobalAvgPool          → 32× 1× 1
      Dropout(0.5)            ← increased from 0.4
      Linear(32→2)
    """
    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,  8,  pool=True),
            ConvBlock(8,  16, pool=True),
            ConvBlock(16, 32, pool=True),
        )
        self.gap        = nn.AdaptiveAvgPool2d(1)
        self.dropout    = nn.Dropout(dropout)
        self.classifier = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.classifier(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad)

# ═══════════════════════════════════════════════════════════════════
# DATA PIPELINE  —  heavy augmentation for small dataset
# ═══════════════════════════════════════════════════════════════════

def build_dataloaders():
    """
    Augmentation strategy for 1680 training samples:

    RandomHorizontalFlip     → eye is symmetric, flip is valid
    RandomVerticalFlip(0.1)  → rare but adds variety
    RandomRotation(15)       → head tilt variation
    ColorJitter(0.4, 0.4)   → lighting variation (bright/dim rooms)
    RandomAffine translate   → eye not always perfectly centred
    RandomAffine scale       → distance from camera varies
    GaussianBlur             → simulates slight focus variation
    RandomErasing            → simulates partial occlusion (glasses,
                               hand near face) — forces model to use
                               overall eyelid shape not just one spot

    These transforms make the 1680 images behave more like 10,000+
    by ensuring the model never sees the exact same image twice.
    """
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),

        # Spatial augmentations
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.10, 0.10),   # shift up to 10% in x and y
            scale=(0.85, 1.15),       # zoom in/out up to 15%
        ),

        # Photometric augmentations
        transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
        ),
        transforms.GaussianBlur(
            kernel_size=3,
            sigma=(0.1, 1.5),         # random blur amount
        ),

        # Tensor + normalise
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),

        # RandomErasing — after ToTensor only
        # Randomly blacks out a rectangular patch (5–20% of image)
        # Forces model to not rely on a single pixel region
        transforms.RandomErasing(
            p=0.3,
            scale=(0.05, 0.20),
            ratio=(0.3, 3.3),
            value=0,
        ),
    ])

    # Val and test: NO augmentation — clean evaluation
    eval_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    train_ds = datasets.ImageFolder(
        str(SPLITS_DIR / "train"), transform=train_transform)
    val_ds   = datasets.ImageFolder(
        str(SPLITS_DIR / "val"),   transform=eval_transform)
    test_ds  = datasets.ImageFolder(
        str(SPLITS_DIR / "test"),  transform=eval_transform)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True)
    val_loader   = DataLoader(
        val_ds,   batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True)
    test_loader  = DataLoader(
        test_ds,  batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True)

    print(f"\n  Dataset       : {SPLITS_DIR.resolve()}")
    print(f"  Classes       : {train_ds.classes}")
    print(f"  class→index   : {train_ds.class_to_idx}")
    print(f"  Train samples : {len(train_ds)}")
    print(f"  Val   samples : {len(val_ds)}")
    print(f"  Test  samples : {len(test_ds)}")
    print(f"  Balance       : 50/50 ✓")

    return train_loader, val_loader, test_loader, train_ds

# ═══════════════════════════════════════════════════════════════════
# TRAIN / EVAL
# ═══════════════════════════════════════════════════════════════════

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()

    total_loss = 0.0
    all_preds  = []
    all_labels = []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if train:
                optimizer.zero_grad()

            logits = model(images)
            loss   = criterion(logits, labels)

            if train:
                loss.backward()
                # Gradient clipping — prevents exploding gradients
                # on small batches with high augmentation
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    n         = len(loader.dataset)
    avg_loss  = total_loss / n
    accuracy  = np.mean(np.array(all_preds) == np.array(all_labels))
    f1        = f1_score(all_labels, all_preds,
                         average="weighted", zero_division=0)
    precision = precision_score(all_labels, all_preds,
                                average="weighted", zero_division=0)
    recall    = recall_score(all_labels, all_preds,
                             average="weighted", zero_division=0)
    return {
        "loss"     : round(avg_loss,  4),
        "accuracy" : round(accuracy,  4),
        "f1"       : round(f1,        4),
        "precision": round(precision, 4),
        "recall"   : round(recall,    4),
        "preds"    : all_preds,
        "labels"   : all_labels,
    }

# ═══════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════

def plot_curves(history, output_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("TinyEyeNet (Small Dataset) — Training Curves",
                 fontsize=13, fontweight="bold")

    specs = [
        ("Loss",     "train_loss", "val_loss", None),
        ("Accuracy", "train_acc",  "val_acc",  (0, 1)),
        ("F1 Score", "train_f1",   "val_f1",   (0, 1)),
    ]
    for ax, (title, tk, vk, ylim) in zip(axes, specs):
        ax.plot(epochs, history[tk], label="Train", color="#2196F3")
        ax.plot(epochs, history[vk], label="Val",   color="#F44336")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)

    plt.tight_layout()
    out = output_dir / "training_curves.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Curves saved → {out.resolve()}")


def plot_confusion_matrix(labels, preds, class_names, output_dir):
    cm  = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im  = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(len(class_names)),
           yticks=range(len(class_names)),
           xticklabels=class_names,
           yticklabels=class_names,
           xlabel="Predicted", ylabel="True",
           title="Confusion Matrix — Test Set")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = output_dir / "confusion_matrix.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix → {out.resolve()}")

# ═══════════════════════════════════════════════════════════════════
# OVERFITTING MONITOR
# ═══════════════════════════════════════════════════════════════════

def check_overfitting(train_acc, val_acc, epoch):
    """
    Prints a warning if the train/val gap is growing.
    Gap > 15% = overfitting. Gap > 25% = severe overfitting.
    """
    gap = train_acc - val_acc
    if gap > 0.25:
        print(f"  ⚠ OVERFIT WARNING ep{epoch}: "
              f"train={train_acc:.3f} val={val_acc:.3f} "
              f"gap={gap:.3f} — consider collecting more data")
    elif gap > 0.15:
        print(f"  ↑ gap={gap:.3f} (mild overfit — normal for small dataset)")

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def train():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*57}")
    print(f"  TinyEyeNet — Small Dataset Training")
    print(f"{'='*57}")
    print(f"  Device        : {device}")
    if device.type == "cuda":
        print(f"  GPU           : {torch.cuda.get_device_name(0)}")

    # ── Data ──────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, train_ds = build_dataloaders()

    # ── Model ─────────────────────────────────────────────────────
    model = TinyEyeNet(num_classes=2, dropout=0.5).to(device)
    print(f"\n  Model         : TinyEyeNet")
    print(f"  Parameters    : {count_params(model):,}")
    print(f"  Dropout       : 0.5  (increased for small dataset)")
    print(f"  Input shape   : 1 × {IMG_SIZE} × {IMG_SIZE}")

    # ── Loss — no class weights needed (50/50 balance) ────────────
    criterion = nn.CrossEntropyLoss()

    # ── Optimiser + Scheduler ─────────────────────────────────────
    optimizer = optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # ── History ───────────────────────────────────────────────────
    history = {k: [] for k in [
        "train_loss", "train_acc", "train_f1",
        "val_loss",   "val_acc",   "val_f1",
    ]}

    best_val_acc      = 0.0
    epochs_no_improve = 0
    best_model_path   = OUTPUT_DIR / "best_eye_model.pth"

    print(f"\n  Epochs        : {NUM_EPOCHS}")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  LR            : {LR}  (cosine decay → 1e-6)")
    print(f"  Weight decay  : {WEIGHT_DECAY}")
    print(f"  Early stop    : patience={PATIENCE}")
    print(f"\n  {'Ep':>4}  {'T-Loss':>7} {'T-Acc':>6} {'T-F1':>6}  "
          f"{'V-Loss':>7} {'V-Acc':>6} {'V-F1':>6}  {'Time':>6}")
    print(f"  {'─'*68}")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_m = run_epoch(model, train_loader, criterion,
                            optimizer, device, train=True)
        val_m   = run_epoch(model, val_loader,   criterion,
                            optimizer, device, train=False)

        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_m["loss"])
        history["train_acc"].append(train_m["accuracy"])
        history["train_f1"].append(train_m["f1"])
        history["val_loss"].append(val_m["loss"])
        history["val_acc"].append(val_m["accuracy"])
        history["val_f1"].append(val_m["f1"])

        improved = val_m["accuracy"] > best_val_acc
        if improved:
            best_val_acc      = val_m["accuracy"]
            epochs_no_improve = 0
            torch.save({
                "epoch"       : int(epoch),
                "model_state" : model.state_dict(),
                "val_acc"     : float(best_val_acc),
                "val_f1"      : float(val_m["f1"]),
                "class_names" : train_ds.classes,
                "img_size"    : int(IMG_SIZE),
            }, best_model_path)
            tag = " ✓"
        else:
            epochs_no_improve += 1
            tag = ""

        print(f"  {epoch:>4}  "
              f"{train_m['loss']:>7.4f} {train_m['accuracy']:>6.4f} "
              f"{train_m['f1']:>6.4f}  "
              f"{val_m['loss']:>7.4f} {val_m['accuracy']:>6.4f} "
              f"{val_m['f1']:>6.4f}  "
              f"{elapsed:>5.1f}s{tag}")

        # Overfitting monitor — prints warning if gap grows too large
        check_overfitting(train_m["accuracy"], val_m["accuracy"], epoch)

        if epochs_no_improve >= PATIENCE:
            print(f"\n  Early stopping at epoch {epoch}.")
            break

    # ── Post-training ─────────────────────────────────────────────
    print(f"\n  Best val accuracy : {best_val_acc:.4f}")
    print(f"  Model saved       → {best_model_path.resolve()}")

    with open(OUTPUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    plot_curves(history, OUTPUT_DIR)

    # ── Test evaluation ───────────────────────────────────────────
    print(f"\n{'─'*57}")
    print("  Test Set Evaluation (best model)")
    print(f"{'─'*57}")

    checkpoint = torch.load(best_model_path, map_location=device,
                            weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test_m = run_epoch(model, test_loader, criterion,
                       optimizer, device, train=False)

    print(f"\n  Accuracy  : {test_m['accuracy']:.4f}")
    print(f"  F1        : {test_m['f1']:.4f}")
    print(f"  Precision : {test_m['precision']:.4f}")
    print(f"  Recall    : {test_m['recall']:.4f}")

    plot_confusion_matrix(test_m["labels"], test_m["preds"],
                          train_ds.classes, OUTPUT_DIR)

    metrics = {
        "best_epoch"    : checkpoint["epoch"],
        "best_val_acc"  : float(best_val_acc),
        "test_accuracy" : test_m["accuracy"],
        "test_f1"       : test_m["f1"],
        "test_precision": test_m["precision"],
        "test_recall"   : test_m["recall"],
        "class_names"   : train_ds.classes,
        "model_params"  : count_params(model),
        "train_samples" : len(train_ds),
        "augmentation"  : "heavy",
    }
    with open(OUTPUT_DIR / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved → {(OUTPUT_DIR/'test_metrics.json').resolve()}")

    # ── Overfitting summary ───────────────────────────────────────
    final_gap = (history["train_acc"][-1] -
                 history["val_acc"][-1])
    print(f"\n  Final train/val gap : {final_gap:.4f}")
    if final_gap < 0.10:
        print("  ✓ No significant overfitting detected.")
    elif final_gap < 0.20:
        print("  ⚠ Mild overfit — acceptable for this dataset size.")
        print("    Consider collecting 200 more images per class.")
    else:
        print("  ✗ Significant overfit — collect more data.")

    print(f"\n{'='*57}")
    print("  Training complete.")
    print(f"  Outputs → {OUTPUT_DIR.resolve()}")
    print(f"{'='*57}\n")


if __name__ == "__main__":
    train()