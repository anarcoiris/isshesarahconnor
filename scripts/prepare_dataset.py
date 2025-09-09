#!/usr/bin/env python3
"""
Limpieza básica y split del dataset.
- elimina archivos corruptos
- convierte imágenes a RGB
- elimina archivos muy pequeños
- elimina duplicados por hash
- organiza en data/final/train|val|test/<label>

Uso:
  python scripts/prepare_dataset.py --raw data/raw --out data/final --train 0.7 --val 0.15 --test 0.15
"""
import argparse
from pathlib import Path
from PIL import Image
import hashlib
import shutil
import random
from tqdm import tqdm


def file_hash(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def is_valid_image(path, min_size=(50,50)):
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img = img.convert('RGB')
            if img.size[0] < min_size[0] or img.size[1] < min_size[1]:
                return False
        return True
    except Exception:
        return False


def prepare(raw_root, out_root, train_ratio, val_ratio, test_ratio, seed=42):
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    assert raw_root.exists(), f"{raw_root} does not exist"
    labels = [p.name for p in raw_root.iterdir() if p.is_dir()]
    print('Found labels:', labels)

    # collect valid images and deduplicate
    hashes = {}
    items_by_label = {lbl: [] for lbl in labels}
    for lbl in labels:
        for p in raw_root.joinpath(lbl).iterdir():
            if not p.is_file():
                continue
            if not is_valid_image(p):
                continue
            h = file_hash(p)
            if h in hashes:
                continue
            hashes[h] = str(p)
            items_by_label[lbl].append(str(p))

    random.seed(seed)
    for lbl, items in items_by_label.items():
        random.shuffle(items)
        n = len(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train = items[:n_train]
        val = items[n_train:n_train+n_val]
        test = items[n_train+n_val:]

        for split_name, split_list in [('train', train), ('val', val), ('test', test)]:
            target_dir = out_root / split_name / lbl
            target_dir.mkdir(parents=True, exist_ok=True)
            for src in split_list:
                dst = target_dir / Path(src).name
                # copy (not move) to preserve raw
                shutil.copy2(src, dst)

    print('Dataset prepared at', out_root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', default='data/raw')
    parser.add_argument('--out', default='data/final')
    parser.add_argument('--train', type=float, default=0.7)
    parser.add_argument('--val', type=float, default=0.15)
    parser.add_argument('--test', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, 'ratios must sum to 1'
    prepare(args.raw, args.out, args.train, args.val, args.test, seed=args.seed)

if __name__ == '__main__':
    main()
