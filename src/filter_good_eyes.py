# filter_good_eyes.py
import cv2, pathlib, shutil, numpy as np
from tqdm import tqdm

RAW_DIR   = pathlib.Path("data/raw/eye")
CLEAN_DIR = pathlib.Path("data/clean/eye")

for cls in ['open', 'closed']:
    src = RAW_DIR   / cls
    dst = CLEAN_DIR / cls
    dst.mkdir(parents=True, exist_ok=True)

    images = list(src.glob('*.jpg')) + list(src.glob('*.png'))
    ok = skipped = 0

    for p in tqdm(images, desc=cls):
        img = cv2.imread(str(p))
        if img is None:
            skipped += 1
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)

        if edges.mean() < 1.0 or gray.std() < 15:
            skipped += 1
            continue

        shutil.copy2(p, dst / p.name)
        ok += 1

    print(f'{cls}: {ok} copied, {skipped} skipped')

print("\nDone. Clean dataset ready at data/clean/eye/")