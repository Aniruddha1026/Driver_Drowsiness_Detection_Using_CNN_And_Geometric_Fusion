# balance_eye_dataset.py
import pathlib, shutil, random

CLEAN_DIR  = pathlib.Path("data/clean/eye")
BALANCED_DIR = pathlib.Path("data/balanced/eye")
RANDOM_SEED  = 42

random.seed(RANDOM_SEED)

# Count closed (minority class)
closed_imgs = list((CLEAN_DIR / 'closed').glob('*.jpg')) + \
              list((CLEAN_DIR / 'closed').glob('*.png'))
n_target = len(closed_imgs)   # 17,469

for cls in ['open', 'closed']:
    src   = CLEAN_DIR    / cls
    dst   = BALANCED_DIR / cls
    dst.mkdir(parents=True, exist_ok=True)

    imgs = list(src.glob('*.jpg')) + list(src.glob('*.png'))
    random.shuffle(imgs)
    selected = imgs[:n_target]   # cap both at 17,469

    for p in selected:
        shutil.copy2(p, dst / p.name)

    print(f'{cls}: {len(selected)} images → {dst}')

print(f'\nBalanced: {n_target} per class = {n_target*2} total')
