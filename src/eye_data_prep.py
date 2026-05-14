"""
eye_data_prep.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Eye Dataset — Filter → STRICT EQUAL BALANCE → Split

Forces EXACTLY equal open and closed images before
splitting so train/val/test are perfectly 50/50.

Steps:
  1. FILTER   remove blank/bad images from raw
  2. BALANCE  undersample majority to exactly match minority
  3. SPLIT    70 / 15 / 15  (perfectly balanced in each split)
  4. VERIFY   PyTorch ImageFolder check

Input:
  data/raw/eye/
      open/
      closed/

Output:
  data/splits/eye/
      train/open/   train/closed/   ← exactly equal
      val/open/     val/closed/     ← exactly equal
      test/open/    test/closed/    ← exactly equal
"""

import cv2
import pathlib
import shutil
import random
import sys
import numpy as np
from tqdm import tqdm

# ═══════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════

RAW_DIR      = pathlib.Path("data/raw/eye3")
FILTERED_DIR = pathlib.Path("data/filtered/eye3")
BALANCED_DIR = pathlib.Path("data/balanced/eye3")
SPLITS_DIR   = pathlib.Path("data/splits/eye3")

CLASSES      = ["open", "closed"]
IMG_SIZE     = 24
SPLIT_RATIOS = (0.70, 0.15, 0.15)
RANDOM_SEED  = 42

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# Quality filter thresholds
MIN_EDGE_DENSITY = 1.0
MIN_STD          = 15.0

# ═══════════════════════════════════════════════════
#  STEP 1 — FILTER
# ═══════════════════════════════════════════════════

def is_valid(gray_img) -> bool:
    edges = cv2.Canny(gray_img, 30, 100)
    if edges.mean() < MIN_EDGE_DENSITY:
        return False
    if gray_img.std() < MIN_STD:
        return False
    return True


def step_filter():
    print("\n── STEP 1: Filtering bad images ─────────────────────")

    counts = {}
    for cls in CLASSES:
        src = RAW_DIR      / cls
        dst = FILTERED_DIR / cls
        dst.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            print(f"\n[ERROR] {src} not found.")
            print("Expected: data/raw/eye/open/  and  data/raw/eye/closed/")
            sys.exit(1)

        images = [p for p in src.iterdir()
                  if p.suffix.lower() in SUPPORTED_EXT]

        ok = bad = corrupt = 0

        for p in tqdm(images, desc=f"  {cls}", unit="img"):
            img = cv2.imread(str(p))
            if img is None:
                corrupt += 1
                continue

            gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    if len(img.shape) == 3 else img)

            if not is_valid(gray):
                bad += 1
                continue

            resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE),
                                 interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dst / (p.stem + ".png")), resized)
            ok += 1

        counts[cls] = ok
        print(f"     {cls:<8}: {ok:>6} kept  |  {bad:>6} removed")

    return counts


# ═══════════════════════════════════════════════════
#  STEP 2 — STRICT EQUAL BALANCE
# ═══════════════════════════════════════════════════

def step_balance(filter_counts: dict):
    print("\n── STEP 2: Strict Equal Balancing ────────────────────")

    random.seed(RANDOM_SEED)

    # Count available images per class
    available = {}
    for cls in CLASSES:
        imgs = list((FILTERED_DIR / cls).glob("*.png"))
        available[cls] = len(imgs)
        print(f"     {cls:<8}: {len(imgs):>6} filtered images available")

    # Target = minority class count (exact match)
    target = min(available.values())
    minority = min(available, key=available.get)
    majority = max(available, key=available.get)
    removed  = available[majority] - target

    print(f"\n     Minority      : {minority} ({target})")
    print(f"     Majority      : {majority} ({available[majority]})")
    print(f"     Removing      : {removed} from '{majority}'")
    print(f"     Final per class: {target}")
    print(f"     Final total    : {target * 2}")

    for cls in CLASSES:
        src = FILTERED_DIR / cls
        dst = BALANCED_DIR / cls
        dst.mkdir(parents=True, exist_ok=True)

        # Clear destination first to avoid stale files
        for old in dst.glob("*.png"):
            old.unlink()

        imgs = list(src.glob("*.png"))
        random.shuffle(imgs)
        selected = imgs[:target]      # take exactly `target` images

        for p in selected:
            shutil.copy2(p, dst / p.name)

        print(f"     {cls:<8}: {len(selected)} images → balanced/eye/{cls}/")

    return target


# ═══════════════════════════════════════════════════
#  STEP 3 — SPLIT
# ═══════════════════════════════════════════════════

def step_split(n_per_class: int):
    print("\n── STEP 3: Splitting train / val / test ──────────────")

    random.seed(RANDOM_SEED)
    train_r, val_r, _ = SPLIT_RATIOS

    # Pre-calculate expected counts for display
    n_train = int(n_per_class * train_r)
    n_val   = int(n_per_class * val_r)
    n_test  = n_per_class - n_train - n_val

    print(f"     Per class: train={n_train}  val={n_val}  test={n_test}")
    print(f"     Total   : train={n_train*2}  val={n_val*2}  test={n_test*2}")

    split_counts = {s: {} for s in ["train", "val", "test"]}

    for cls in CLASSES:
        src = BALANCED_DIR / cls
        if not src.exists():
            print(f"  [WARN] {src} not found — skipping.")
            continue

        images = sorted(list(src.glob("*.png")))
        random.shuffle(images)

        splits = {
            "train": images[:n_train],
            "val"  : images[n_train : n_train + n_val],
            "test" : images[n_train + n_val:],
        }

        for split_name, files in splits.items():
            dst = SPLITS_DIR / split_name / cls
            dst.mkdir(parents=True, exist_ok=True)

            # Clear old files
            for old in dst.glob("*.png"):
                old.unlink()

            for f in files:
                shutil.copy2(f, dst / f.name)
            split_counts[split_name][cls] = len(files)

    # Print final table
    print(f"\n  {'Split':<8} {'open':>8} {'closed':>8} {'total':>8} {'balance':>10}")
    print(f"  {'─'*48}")
    for split in ["train", "val", "test"]:
        o = split_counts[split].get("open",   0)
        c = split_counts[split].get("closed", 0)
        bal = "✓ 50/50" if o == c else f"! {o}/{c}"
        print(f"  {split:<8} {o:>8} {c:>8} {o+c:>8} {bal:>10}")

    return split_counts


# ═══════════════════════════════════════════════════
#  STEP 4 — VERIFY
# ═══════════════════════════════════════════════════

def step_verify(split_counts: dict, n_per_class: int):
    print("\n── STEP 4: PyTorch Readiness Check ───────────────────")

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
            print("  [SKIP] No splits found — run previous steps first.")
            return

        train_ds = datasets.ImageFolder(
            str(train_dir), transform=train_transform)
        val_ds   = datasets.ImageFolder(
            str(SPLITS_DIR / "val"),  transform=eval_transform)
        test_ds  = datasets.ImageFolder(
            str(SPLITS_DIR / "test"), transform=eval_transform)

        print(f"\n  Classes       : {train_ds.classes}")
        print(f"  class→index   : {train_ds.class_to_idx}")
        print(f"  Train samples : {len(train_ds)}")
        print(f"  Val   samples : {len(val_ds)}")
        print(f"  Test  samples : {len(test_ds)}")

        # Verify tensor shape and range
        tensor, label = train_ds[0]
        print(f"\n  Sample shape  : {tensor.shape}")
        print(f"  Pixel range   : [{tensor.min():.3f}, {tensor.max():.3f}]")
        print(f"  Label         : {label} ({train_ds.classes[label]})")

        # Confirm perfect balance
        train_o = split_counts["train"].get("open",   0)
        train_c = split_counts["train"].get("closed", 0)

        if train_o == train_c:
            print(f"\n  ✓ Perfect 50/50 balance — "
                  f"no class weights needed in training.")
            print(f"  Use standard: criterion = nn.CrossEntropyLoss()")
        else:
            ratio = max(train_o, train_c) / (min(train_o, train_c) + 1e-6)
            print(f"\n  ⚠ Slight imbalance ({ratio:.2f}x) — "
                  f"check split logic.")

        print(f"\n  ✓ Eye dataset ready for Week 3 training.")
        print(f"  Path: {SPLITS_DIR.resolve()}")

    except ImportError:
        print("  torchvision not installed: pip install torchvision")


# ═══════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════

def main():
    print("=" * 52)
    print("  Eye Dataset Prep — Strict 50/50 Balance + Split")
    print("=" * 52)

    if not RAW_DIR.exists():
        print(f"\n[ERROR] {RAW_DIR.resolve()} not found.")
        print("\nCreate:")
        print("  data/raw/eye/open/    ← open eye images")
        print("  data/raw/eye/closed/  ← closed eye images")
        sys.exit(1)

    filter_counts = step_filter()
    n_per_class   = step_balance(filter_counts)
    split_counts  = step_split(n_per_class)
    step_verify(split_counts, n_per_class)

    print("\n" + "=" * 52)
    print("  Done. Perfectly balanced splits ready.")
    print(f"  {SPLITS_DIR.resolve()}")
    print("=" * 52 + "\n")


if __name__ == "__main__":
    main()