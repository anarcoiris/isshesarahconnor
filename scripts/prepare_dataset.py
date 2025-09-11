#!/usr/bin/env python3
"""
prepare_dataset.py (mejorado)

Limpieza básica y split del dataset:
- elimina archivos corruptos
- convierte imágenes a RGB (mitiga warnings de paleta/transparencia)
- elimina archivos muy pequeños
- elimina duplicados por hash (y opcionalmente por phash si imagehash está instalado)
- organiza en data/final/train|val|test/<label>

Uso:
  python scripts/prepare_dataset.py --raw data/raw --out data/final --train 0.7 --val 0.15 --test 0.15
"""
from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image, ImageFile
import hashlib
import shutil
import random
from tqdm import tqdm
import sys
import os

ImageFile.LOAD_TRUNCATED_IMAGES = True  # permitir imágenes parcialmente corruptas si es posible

# Optional perceptual hashing
try:
    import imagehash  # type: ignore
    HAS_IMAGEHASH = True
except Exception:
    HAS_IMAGEHASH = False

# ---------------- helpers ----------------
def file_hash(path: Path, chunk_size: int = 8192) -> str:
    h = hashlib.md5()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()

def safe_open_convert_rgb(path: Path):
    """
    Abre la imagen, convierte a RGB y devuelve PIL.Image.
    Maneja imágenes con paleta/transparency y distintos modos (L, P, RGBA, CMYK).
    """
    try:
        with Image.open(path) as im:
            # Force load to catch errors early
            im.load()
            # Convert palette/transparency to RGBA first, then to RGB
            if im.mode == 'P':
                im = im.convert('RGBA')
            elif im.mode == 'LA':
                im = im.convert('RGBA')
            elif im.mode == 'CMYK':
                im = im.convert('RGB')
            elif im.mode == 'RGBA':
                # For RGBA, compose over white background to avoid alpha channel problems
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.split()[3])  # 3 = alpha
                im = bg
            else:
                im = im.convert('RGB')
            return im
    except Exception as e:
        return None

def is_valid_image(path: Path, min_size=(50,50)) -> bool:
    """
    Verifica que la imagen sea válida y tenga al menos min_size.
    """
    im = safe_open_convert_rgb(path)
    if im is None:
        return False
    w,h = im.size
    if w < min_size[0] or h < min_size[1]:
        return False
    return True

def canonical_label(name: str) -> str:
    """
    Normaliza nombres de carpeta/etiqueta, por ejemplo "pos_raw" -> "pos".
    Ajusta según patrones frecuentes.
    """
    name = name.strip()
    # eliminar sufijos/prefijos típicos generados por crawlers
    for suf in ['_raw', '-raw', '.raw', '_images', '-images', '_img']:
        if name.endswith(suf):
            name = name[: -len(suf)]
    # también manejar mayúsculas
    return name.lower()

def ensure_unique_filename(dest_dir: Path, original_name: str, ext: str, max_tries: int = 1000) -> Path:
    """
    Si existe colisión, añade un sufijo incremental o usa hash.
    """
    candidate = dest_dir / f"{original_name}.{ext}"
    if not candidate.exists():
        return candidate
    base = original_name
    for i in range(1, max_tries):
        candidate = dest_dir / f"{base}_{i}.{ext}"
        if not candidate.exists():
            return candidate
    # fallback: use uuid-ish hash
    import time
    suffix = hashlib.md5((original_name + str(time.time())).encode()).hexdigest()[:8]
    return dest_dir / f"{base}_{suffix}.{ext}"

# ---------------- main logic ----------------
def prepare(raw_root: Path, out_root: Path, train_ratio: float, val_ratio: float, test_ratio: float,
            min_size=(50,50), dedupe_phash: bool = False, phash_threshold: int = 6, seed: int = 42):
    assert raw_root.exists(), f"{raw_root} does not exist"
    # collect label dirs recursively (only first level)
    labels_raw = [p.name for p in raw_root.iterdir() if p.is_dir()]
    labels_canonical_map = {}
    for lbl in labels_raw:
        labels_canonical_map[lbl] = canonical_label(lbl)
    # group by canonical label
    labels_by_canon = {}
    for orig, canon in labels_canonical_map.items():
        labels_by_canon.setdefault(canon, []).append(orig)
    print("Found label folders (raw -> canonical):")
    for canon, raws in labels_by_canon.items():
        print(f"  {canon}: {raws}")

    # gather valid files and deduplicate by MD5
    hashes = {}
    items_by_label = {canon: [] for canon in labels_by_canon}
    total_examined = 0
    total_valid = 0
    for canon_label, raw_dirs in labels_by_canon.items():
        for raw_dir in raw_dirs:
            pdir = raw_root / raw_dir
            for p in pdir.rglob("*"):
                if not p.is_file():
                    continue
                total_examined += 1
                # quick ext filter
                ext = p.suffix.lower()
                if ext not in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tif', '.tiff']:
                    # still try to open (some images have no extension)
                    pass
                if not is_valid_image(p, min_size=min_size):
                    continue
                # binary hash
                try:
                    h = file_hash(p)
                except Exception:
                    continue
                if h in hashes:
                    # duplicate file exact copy: skip
                    continue
                hashes[h] = str(p)
                items_by_label[canon_label].append((p, h))
                total_valid += 1

    print(f"Examined {total_examined} files, found {total_valid} valid (unique by MD5).")

    # optionally dedupe by perceptual hash (phash)
    if dedupe_phash and HAS_IMAGEHASH:
        print("Performing perceptual deduplication using imagehash (phash)...")
        phash_map = {}  # phash -> (path, md5)
        kept = {}
        for canon, items in items_by_label.items():
            kept[canon] = []
            for p, md5 in tqdm(items, desc=f"phash {canon}"):
                try:
                    im = safe_open_convert_rgb(p)
                    if im is None:
                        continue
                    ph = imagehash.phash(im)
                except Exception:
                    # skip if phash fails
                    kept[canon].append((p, md5))
                    continue
                # compare with existing
                collided = False
                for existing_ph, (ep, eh) in phash_map.items():
                    # hamming distance:
                    dist = ph - existing_ph
                    if dist <= phash_threshold:
                        collided = True
                        break
                if not collided:
                    phash_map[ph] = (p, md5)
                    kept[canon].append((p, md5))
            items_by_label[canon] = kept[canon]
        # recompute total_valid
        total_valid = sum(len(v) for v in items_by_label.values())
        print(f"After perceptual dedupe: {total_valid} images remain.")
    else:
        if dedupe_phash and not HAS_IMAGEHASH:
            print("imagehash no está instalado; saltando dedupe perceptual. Para activarlo: pip install ImageHash")

    # prepare output dirs and splits
    random.seed(seed)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for canon, items in items_by_label.items():
        n = len(items)
        if n == 0:
            print(f"[warning] No images for label '{canon}', skipping.")
            continue
        random.shuffle(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # ensure at least one in train if possible
        if n_train == 0 and n >= 1:
            n_train = 1
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)  # keep at least 1 for test if possible
        train = items[:n_train]
        val = items[n_train:n_train + n_val]
        test = items[n_train + n_val:]
        splits = [('train', train), ('val', val), ('test', test)]
        summary[canon] = {'total': n, 'train': len(train), 'val': len(val), 'test': len(test)}
        for split_name, split_list in splits:
            target_dir = out_root / split_name / canon
            target_dir.mkdir(parents=True, exist_ok=True)
            for src_path, md5 in split_list:
                # preserve extension if possible
                ext = src_path.suffix.lower().lstrip('.') or 'jpg'
                base_name = src_path.stem
                # sanitize base_name
                base_name = "".join(c for c in base_name if c.isalnum() or c in ('-', '_')).strip()[:60] or md5[:8]
                dest_path = ensure_unique_filename(target_dir, base_name, ext)
                # copy only if destination not exists or different
                try:
                    if dest_path.exists():
                        # compare content
                        existing_hash = file_hash(dest_path)
                        if existing_hash == md5:
                            continue  # already copied
                    shutil.copy2(src_path, dest_path)
                except Exception as e:
                    # fallback: try rewriting as RGB JPEG to avoid weird formats causing problems later
                    try:
                        im = safe_open_convert_rgb(src_path)
                        if im is None:
                            continue
                        # save as jpg
                        fallback_name = ensure_unique_filename(target_dir, base_name, 'jpg')
                        im.save(fallback_name, format='JPEG', quality=92)
                    except Exception:
                        continue

    # print summary
    print("Dataset prepared at", out_root)
    print("Summary by label:")
    for k,v in summary.items():
        print(f"  {k}: total={v['total']}, train={v['train']}, val={v['val']}, test={v['test']}")

# ---------------- CLI ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', default='data/raw')
    parser.add_argument('--out', default='data/final')
    parser.add_argument('--train', type=float, default=0.7)
    parser.add_argument('--val', type=float, default=0.15)
    parser.add_argument('--test', type=float, default=0.15)
    parser.add_argument('--min_size', type=int, nargs=2, default=(50,50), help="Min width height")
    parser.add_argument('--dedupe_phash', action='store_true', help='Intentar eliminar imagenes similares (requiere imagehash)')
    parser.add_argument('--phash_threshold', type=int, default=6, help='Hamming distance threshold for phash dedupe (lower = stricter)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        parser.error("train+val+test must sum to 1.0")

    raw_root = Path(args.raw)
    out_root = Path(args.out)
    prepare(raw_root, out_root, args.train, args.val, args.test,
            min_size=tuple(args.min_size), dedupe_phash=args.dedupe_phash,
            phash_threshold=args.phash_threshold, seed=args.seed)

if __name__ == '__main__':
    main()
