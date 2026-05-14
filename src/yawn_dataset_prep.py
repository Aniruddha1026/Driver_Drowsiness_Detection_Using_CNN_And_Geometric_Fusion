"""
Week 2 — Yawn Dataset Preparation (Pre-Cropped Images)
WITH brightness-based quality filter to reject bad crops.

Filter logic:
  yawn images   → reject if mean brightness > 180  (too bright = no open mouth)
  no_yawn images→ reject if mean brightness < 40   (too dark  = not a mouth)
  both classes  → reject if std < 15               (blank/uniform = bad crop)
"""

import cv2
import numpy as np
import pathlib
import shutil
import random
import sys
from tqdm import tqdm

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

RAW_DIR       = pathlib.Path("data/raw")
PROCESSED_DIR = pathlib.Path("data/processed/mouth")
SPLITS_DIR    = pathlib.Path("data/splits/mouth")
REJECTED_DIR  = pathlib.Path("data/rejected/mouth")

CLASSES       = ["yawn", "no_yawn"]
IMG_SIZE      = 64
SPLIT_RATIOS  = (0.70, 0.15, 0.15)
RANDOM_SEED   = 42
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# ── Quality filter thresholds ─────────────────────────────────────
# Yawn:    open mouth = dark cavity  → reject if TOO BRIGHT
# No-yawn: closed lips = mid-bright  → reject if TOO DARK
# Both:    uniform/blank             → reject if std too low
FILTER = {
    "yawn"   : {"min_mean": 10,  "max_mean": 180, "min_std": 15},
    "no_yawn": {"min_mean": 40,  "max_mean": 230, "min_std": 15},
}

# ═══════════════════════════════════════════════════════════════════
# QUALITY FILTER
# ═══════════════════════════════════════════════════════════════════

def is_valid(gray_img, cls):
    """
    Returns (True, '') if image passes quality check.
    Returns (False, reason) if it should be rejected.
    """
    mean = gray_img.mean()
    std  = gray_img.std()
    f    = FILTER[cls]

    if std < f["min_std"]:
        return False, f"blank/uniform (std={std:.1f})"
    if mean < f["min_mean"]:
        return False, f"too dark (mean={mean:.1f})"
    if mean > f["max_mean"]:
        return False, f"too bright (mean={mean:.1f})"
    return True, ""

# ═══════════════════════════════════════════════════════════════════
# STEP 1 — PREPROCESS + FILTER
# ═══════════════════════════════════════════════════════════════════

def preprocess():
    print("\n── STEP 1: Preprocessing + quality filtering ───────────")

    stats = {}

    for cls in CLASSES:
        src_dir      = RAW_DIR / cls
        dst_dir      = PROCESSED_DIR / cls
        rejected_dir = REJECTED_DIR / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        rejected_dir.mkdir(parents=True, exist_ok=True)

        if not src_dir.exists():
            print(f"  [WARN] Not found: {src_dir} — skipping.")
            stats[cls] = {"ok": 0, "rejected": 0, "failed": 0}
            continue

        images = [p for p in src_dir.iterdir()
                  if p.suffix.lower() in SUPPORTED_EXT]

        ok = rejected = failed = 0

        for img_path in tqdm(images, desc=f"  {cls}", unit="img"):
            img = cv2.imread(str(img_path))
            if img is None:
                failed += 1
                continue

            # Grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) \
                   if len(img.shape) == 3 else img

            # Resize
            resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE),
                                 interpolation=cv2.INTER_AREA)

            # Quality check
            valid, reason = is_valid(resized, cls)
            if not valid:
                shutil.copy2(img_path, rejected_dir / img_path.name)
                rejected += 1
                continue

            cv2.imwrite(str(dst_dir / (img_path.stem + ".png")), resized)
            ok += 1

        stats[cls] = {"ok": ok, "rejected": rejected, "failed": failed}
        print(f"     {cls}: {ok} saved | {rejected} rejected | {failed} unreadable")
        if rejected > 0:
            print(f"     → Rejected saved to: {(REJECTED_DIR/cls).resolve()}")
            print(f"       Review them to confirm the filter is working correctly.")

    return stats

# ═══════════════════════════════════════════════════════════════════
# STEP 2 — SPLIT
# ═══════════════════════════════════════════════════════════════════

def split_dataset():
    print("\n── STEP 2: Splitting into train / val / test ───────────")

    random.seed(RANDOM_SEED)
    train_r, val_r, _ = SPLIT_RATIOS
    split_counts = {s: {} for s in ["train", "val", "test"]}

    for cls in CLASSES:
        src_dir = PROCESSED_DIR / cls
        if not src_dir.exists():
            continue

        images = sorted([p for p in src_dir.iterdir()
                         if p.suffix.lower() == ".png"])
        random.shuffle(images)

        n       = len(images)
        n_train = int(n * train_r)
        n_val   = int(n * val_r)

        splits = {
            "train": images[:n_train],
            "val"  : images[n_train:n_train + n_val],
            "test" : images[n_train + n_val:],
        }

        for split_name, files in splits.items():
            dst = SPLITS_DIR / split_name / cls
            dst.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, dst / f.name)
            split_counts[split_name][cls] = len(files)

    print(f"\n  {'Split':<8} {'yawn':>8} {'no_yawn':>10} {'total':>8}")
    print(f"  {'─'*38}")
    for split in ["train", "val", "test"]:
        y = split_counts[split].get("yawn",    0)
        n = split_counts[split].get("no_yawn", 0)
        print(f"  {split:<8} {y:>8} {n:>10} {y+n:>8}")

    return split_counts

# ═══════════════════════════════════════════════════════════════════
# STEP 3 — BALANCE REPORT
# ═══════════════════════════════════════════════════════════════════

def balance_report(split_counts):
    print("\n── STEP 3: Class Balance Report ────────────────────────")

    train = split_counts.get("train", {})
    y = train.get("yawn",    0)
    n = train.get("no_yawn", 0)

    if y + n == 0:
        print("  No training samples found.")
        return

    ratio    = max(y, n) / (min(y, n) + 1e-6)
    minority = "yawn" if y < n else "no_yawn"

    print(f"\n  yawn: {y}  |  no_yawn: {n}  |  ratio: {ratio:.2f}x")

    if ratio <= 1.5:
        print("  ✓ Well balanced — no action needed.")
    elif ratio <= 2.5:
        print(f"  ⚠ Mild imbalance — minority: '{minority}'")
        _weight_snippet(y, n)
    else:
        print(f"  ✗ Severe imbalance — minority: '{minority}'")
        _weight_snippet(y, n)


def _weight_snippet(y, n):
    total    = y + n
    w_yawn   = round(total / (2 * y   + 1e-6), 4)
    w_noyawn = round(total / (2 * n + 1e-6), 4)
    print(f"""
    Paste into Week 3 training script:
    ───────────────────────────────────────────
    class_weights = torch.tensor([{w_yawn}, {w_noyawn}])
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    ───────────────────────────────────────────""")

# ═══════════════════════════════════════════════════════════════════
# STEP 4 — PYTORCH READINESS
# ═══════════════════════════════════════════════════════════════════

def verify_pytorch_readiness():
    print("\n── STEP 4: PyTorch Readiness Check ─────────────────────")
    try:
        from torchvision import datasets, transforms
        import torch

        train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        eval_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

        train_dir = SPLITS_DIR / "train"
        if not train_dir.exists():
            print("  [SKIP] No splits found yet — run steps 1 & 2 first.")
            return

        train_ds = datasets.ImageFolder(str(train_dir), transform=train_transform)
        val_ds   = datasets.ImageFolder(str(SPLITS_DIR / "val"),  transform=eval_transform)
        test_ds  = datasets.ImageFolder(str(SPLITS_DIR / "test"), transform=eval_transform)

        print(f"\n  Classes       : {train_ds.classes}")
        print(f"  class→index   : {train_ds.class_to_idx}")
        print(f"  Train samples : {len(train_ds)}")
        print(f"  Val   samples : {len(val_ds)}")
        print(f"  Test  samples : {len(test_ds)}")

        tensor, label = train_ds[0]
        print(f"\n  Sample shape  : {tensor.shape}")
        print(f"  Pixel range   : [{tensor.min():.3f}, {tensor.max():.3f}]")
        print(f"  Label         : {label} ({train_ds.classes[label]})")
        print("\n  ✓ Ready for Week 3 training.")

    except ImportError:
        print("  torchvision not installed: pip install torchvision")

# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 55)
    print("  Week 2 — Yawn Data Prep (Pre-Cropped + Filtered)")
    print("=" * 55)

    if not RAW_DIR.exists():
        print(f"\n[ERROR] {RAW_DIR.resolve()} not found.")
        print("Create:  data/raw/mouth/yawn/   and   data/raw/mouth/no_yawn/")
        sys.exit(1)

    preprocess()
    counts = split_dataset()
    balance_report(counts)
    verify_pytorch_readiness()

    print("\n" + "=" * 55)
    print("  Done. Check data/rejected/ for filtered-out images.")
    print("=" * 55)


if __name__ == "__main__":
    main()