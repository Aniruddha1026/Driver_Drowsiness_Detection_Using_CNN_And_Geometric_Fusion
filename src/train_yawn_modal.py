"""
Week 3 — Lightweight CNN Training: Yawn Detection
Project: Real-time Driver Drowsiness Detection

Architecture: TinyMouthNet
  - 4 conv blocks (16 → 32 → 64 → 128 filters)
  - Global Average Pooling
  - 2-class output (no_yawn=0, yawn=1)
  - Input: 1 × 64 × 64 grayscale

Why 64×64 and 4 blocks (vs eye model's 24×24 and 3 blocks):
  Mouth anatomy is more complex than eyelid state.
  Jaw drop distance, tooth visibility, lip curl, and tongue
  position all contribute to the yawn signal — more spatial
  detail is needed, so a larger input and deeper network
  is justified while still staying lightweight.

Features:
  - GPU / CPU auto-detection
  - Best model checkpoint by val accuracy
  - Full per-epoch metrics: loss, acc, F1, precision, recall
  - Training curves + confusion matrix saved as PNG
  - Early stopping (patience=10)
  - Class-weighted loss (handles yawn/no_yawn imbalance)
  - Reproducible (fixed seed)
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

SPLITS_DIR  = pathlib.Path("data/splits/mouth")
OUTPUT_DIR  = pathlib.Path("modals/mouth")

IMG_SIZE    = 64          # mouth model: 64×64
BATCH_SIZE  = 32
NUM_EPOCHS  = 60          # slightly more epochs — mouth is harder
LR          = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE    = 10
SEED        = 42

# ImageFolder assigns labels alphabetically:
# 'no_yawn'=0, 'yawn'=1
CLASS_NAMES = ["no_yawn", "yawn"]

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
# MODEL — TinyMouthNet
# ═══════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """Conv(3×3) → BN → ReLU → MaxPool(2×2)"""
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


class TinyMouthNet(nn.Module):
    """
    Lightweight CNN for 1×64×64 mouth images.

    Spatial dimensions through the network:
      Input              →  1 × 64 × 64
      ConvBlock(1→16,  pool) → 16 × 32 × 32
      ConvBlock(16→32, pool) → 32 × 16 × 16
      ConvBlock(32→64, pool) → 64 ×  8 ×  8
      ConvBlock(64→128,pool) →128 ×  4 ×  4
      GlobalAvgPool          →128 ×  1 ×  1
      Flatten                →128
      Dropout(0.5)
      Linear(128→64)  + ReLU
      Dropout(0.3)
      Linear(64→2)           → 2 logits

    Why deeper than TinyEyeNet:
      Eye open/closed is a binary texture difference (eyelid position).
      Yawn detection requires understanding spatial structure:
        - jaw drop distance
        - tooth/tongue visibility
        - lip separation geometry
      An extra conv block doubles the receptive field to capture
      these larger structural patterns, while GAP keeps it inference-fast.

    Total params: ~115 K — still lightweight for real-time use.
    """
    def __init__(self, num_classes=2, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,   16,  pool=True),
            ConvBlock(16,  32,  pool=True),
            ConvBlock(32,  64,  pool=True),
            ConvBlock(64,  128, pool=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)   # (B, 128)
        return self.classifier(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad)

# ═══════════════════════════════════════════════════════════════════
# DATA PIPELINE
# ═══════════════════════════════════════════════════════════════════

def build_dataloaders():
    """
    Training augmentations are more aggressive than the eye model
    because mouth images vary more in practice:
      - drivers tilt head while yawning
      - jaw drop angle varies per person
      - lighting changes affect tooth visibility
    ColorJitter range is wider (0.4) to simulate dashboard glare
    and low-light night driving conditions.
    """
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

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

    return train_loader, val_loader, test_loader, train_ds


def compute_class_weights(dataset, device):
    """
    Inverse-frequency weights so the minority class (usually 'yawn')
    receives a proportionally higher loss penalty.

    Formula: weight[c] = total / (num_classes × count[c])
    """
    counts  = np.bincount([label for _, label in dataset.samples])
    total   = counts.sum()
    weights = total / (len(counts) * counts.astype(np.float32))
    tensor  = torch.tensor(weights, dtype=torch.float32).to(device)

    weight_dict = {cls: round(w, 3)
                   for cls, w in zip(dataset.classes,
                                     tensor.cpu().tolist())}
    print(f"\n  Class counts  : { dict(zip(dataset.classes, counts.tolist())) }")
    print(f"  Class weights : {weight_dict}")

    imbalance = counts.max() / (counts.min() + 1e-6)
    if imbalance > 2.5:
        print(f"  ⚠ Imbalance ratio {imbalance:.1f}x — "
              f"weighted loss is important here.")
    else:
        print(f"  ✓ Imbalance ratio {imbalance:.1f}x — dataset is reasonably balanced.")

    return tensor

# ═══════════════════════════════════════════════════════════════════
# TRAIN / EVAL ONE EPOCH
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

    # Per-class F1 for yawn vs no_yawn visibility
    f1_per_class = f1_score(all_labels, all_preds,
                            average=None, zero_division=0)

    return {
        "loss"        : round(avg_loss,  4),
        "accuracy"    : round(accuracy,  4),
        "f1"          : round(f1,        4),
        "precision"   : round(precision, 4),
        "recall"      : round(recall,    4),
        "f1_per_class": [round(v, 4) for v in f1_per_class.tolist()],
        "preds"       : all_preds,
        "labels"      : all_labels,
    }

# ═══════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════

def plot_curves(history, output_dir):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("TinyMouthNet — Training Curves",
                 fontsize=13, fontweight="bold")

    specs = [
        ("Loss",     "train_loss", "val_loss", None),
        ("Accuracy", "train_acc",  "val_acc",  (0, 1)),
        ("F1 Score", "train_f1",   "val_f1",   (0, 1)),
    ]
    for ax, (title, tr_key, vl_key, ylim) in zip(axes, specs):
        ax.plot(epochs, history[tr_key], label="Train", color="#2196F3")
        ax.plot(epochs, history[vl_key], label="Val",   color="#F44336")
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
    print(f"\n  Curves saved        → {out.resolve()}")


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
    print(f"  Confusion matrix    → {out.resolve()}")


def plot_per_class_f1(history, class_names, output_dir):
    """
    Plot F1 per class over epochs — important for yawn detection
    because we care more about yawn recall than overall accuracy.
    Missing a yawn (false negative) is more dangerous than a
    false alarm (false positive).
    """
    epochs = range(1, len(history["val_f1_per_class"]) + 1)
    fig, ax = plt.subplots(figsize=(8, 4))

    colours = ["#4CAF50", "#FF5722"]
    for i, (cls, col) in enumerate(zip(class_names, colours)):
        vals = [e[i] for e in history["val_f1_per_class"]]
        ax.plot(epochs, vals, label=f"Val F1 — {cls}", color=col)

    ax.set_title("Per-Class F1 Score (Validation)")
    ax.set_xlabel("Epoch")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = output_dir / "per_class_f1.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Per-class F1 chart  → {out.resolve()}")

# ═══════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════

def train():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*57}")
    print(f"  Week 3 — TinyMouthNet Yawn Detection Training")
    print(f"{'='*57}")
    print(f"  Device        : {device}")
    if device.type == "cuda":
        print(f"  GPU           : {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM          : {mem:.1f} GB")

    # ── Data ──────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, train_ds = build_dataloaders()
    class_weights = compute_class_weights(train_ds, device)

    # ── Model ─────────────────────────────────────────────────────
    model = TinyMouthNet(num_classes=2, dropout=0.5).to(device)
    print(f"\n  Model         : TinyMouthNet")
    print(f"  Parameters    : {count_params(model):,}")
    print(f"  Input shape   : 1 × {IMG_SIZE} × {IMG_SIZE}")

    # ── Loss / Optimiser / Scheduler ──────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(),
                           lr=LR, weight_decay=WEIGHT_DECAY)
    # CosineAnnealing smoothly decays LR — better than ReduceLROnPlateau
    # for mouth model which needs finer convergence than the eye model
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-5
    )

    # ── History ───────────────────────────────────────────────────
    history = {k: [] for k in [
        "train_loss", "train_acc", "train_f1",
        "val_loss",   "val_acc",   "val_f1",
        "val_f1_per_class",
    ]}

    best_val_acc      = 0.0
    best_val_f1_yawn  = 0.0   # track yawn-class F1 separately
    epochs_no_improve = 0
    best_model_path   = OUTPUT_DIR / "best_mouth_model.pth"

    print(f"\n  Epochs        : {NUM_EPOCHS}")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  LR schedule   : CosineAnnealing (LR {LR} → 1e-5)")
    print(f"  Early stop    : patience={PATIENCE}")
    print(f"\n  {'Ep':>4}  {'T-Loss':>7} {'T-Acc':>6} {'T-F1':>6}  "
          f"{'V-Loss':>7} {'V-Acc':>6} {'V-F1':>6} "
          f"{'F1-yawn':>8}  {'Time':>6}")
    print(f"  {'─'*76}")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_m = run_epoch(model, train_loader, criterion,
                            optimizer, device, train=True)
        val_m   = run_epoch(model, val_loader,   criterion,
                            optimizer, device, train=False)

        scheduler.step()
        elapsed = time.time() - t0

        # yawn class F1 — index 1 (alphabetical: no_yawn=0, yawn=1)
        f1_yawn = val_m["f1_per_class"][1] \
                  if len(val_m["f1_per_class"]) > 1 else 0.0

        # Record
        history["train_loss"].append(train_m["loss"])
        history["train_acc"].append(train_m["accuracy"])
        history["train_f1"].append(train_m["f1"])
        history["val_loss"].append(val_m["loss"])
        history["val_acc"].append(val_m["accuracy"])
        history["val_f1"].append(val_m["f1"])
        history["val_f1_per_class"].append(val_m["f1_per_class"])

        # Save best on val accuracy
        improved = val_m["accuracy"] > best_val_acc
        if improved:
            best_val_acc     = val_m["accuracy"]
            best_val_f1_yawn = f1_yawn
            epochs_no_improve = 0
            torch.save({
                "epoch"       : int(epoch),
                "model_state" : model.state_dict(),
                "val_acc"     : float(best_val_acc),
                "val_f1"      : float(val_m["f1"]),
                "val_f1_yawn" : float(f1_yawn),
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
              f"{val_m['f1']:>6.4f} {f1_yawn:>8.4f}  "
              f"{elapsed:>5.1f}s{tag}")

        if epochs_no_improve >= PATIENCE:
            print(f"\n  Early stopping triggered at epoch {epoch}.")
            break

    # ── Post-training ─────────────────────────────────────────────
    print(f"\n  Best val accuracy     : {best_val_acc:.4f}")
    print(f"  Best val F1 (yawn)    : {best_val_f1_yawn:.4f}")
    print(f"  Model saved           → {best_model_path.resolve()}")

    history_path = OUTPUT_DIR / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    plot_curves(history, OUTPUT_DIR)
    plot_per_class_f1(history, train_ds.classes, OUTPUT_DIR)

    # ── Test evaluation ───────────────────────────────────────────
    print(f"\n{'─'*57}")
    print("  Test Set Evaluation (best model)")
    print(f"{'─'*57}")

    checkpoint = torch.load(best_model_path, map_location=device,
                            weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    test_m = run_epoch(model, test_loader, criterion,
                       optimizer, device, train=False)

    f1_per = test_m["f1_per_class"]
    print(f"\n  Overall:")
    print(f"    Accuracy  : {test_m['accuracy']:.4f}")
    print(f"    F1        : {test_m['f1']:.4f}")
    print(f"    Precision : {test_m['precision']:.4f}")
    print(f"    Recall    : {test_m['recall']:.4f}")
    print(f"\n  Per-class F1:")
    for cls, score in zip(train_ds.classes, f1_per):
        marker = " ← key metric" if cls == "yawn" else ""
        print(f"    {cls:<10}: {score:.4f}{marker}")

    plot_confusion_matrix(test_m["labels"], test_m["preds"],
                          train_ds.classes, OUTPUT_DIR)

    metrics = {
        "best_epoch"        : checkpoint["epoch"],
        "best_val_acc"      : best_val_acc,
        "best_val_f1_yawn"  : best_val_f1_yawn,
        "test_accuracy"     : test_m["accuracy"],
        "test_f1"           : test_m["f1"],
        "test_precision"    : test_m["precision"],
        "test_recall"       : test_m["recall"],
        "test_f1_no_yawn"   : f1_per[0] if len(f1_per) > 0 else None,
        "test_f1_yawn"      : f1_per[1] if len(f1_per) > 1 else None,
        "model_params"      : count_params(model),
        "class_names"       : train_ds.classes,
    }
    metrics_path = OUTPUT_DIR / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Metrics saved       → {metrics_path.resolve()}")

    print(f"\n{'='*57}")
    print("  Training complete.")
    print(f"  All outputs → {OUTPUT_DIR.resolve()}")
    print(f"{'='*57}")


if __name__ == "__main__":
    train()