#!/usr/bin/env python3
"""
download_and_clean_images.py

Uso ejemplo:
python scripts/download_and_clean_images.py --pos "Sarah Connor, terminator" --neg "middle age woman" --num 1000 --engine google

Descripción:
 - descarga con icrawler o SerpAPI
 - valida/normaliza imágenes (RGB, tamaño mínimo)
 - calcula phash perceptual y elimina duplicados (hamming threshold)
 - guarda metadatos JSON por imagen y un resumen CSV
"""
from __future__ import annotations

import argparse
import os
import json
import time
import math
import logging
from pathlib import Path
from urllib.parse import urlsplit
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image, UnidentifiedImageError
import imagehash
from tqdm import tqdm
import pandas as pd

# opcional: face detection
try:
    import face_recognition  # type: ignore
    _HAS_FACE = True
except Exception:
    _HAS_FACE = False

# icrawler optional
try:
    from icrawler.builtin import GoogleImageCrawler, BingImageCrawler  # type: ignore
    _HAS_ICRAWLER = True
except Exception:
    _HAS_ICRAWLER = False

# serpapi optional
try:
    from serpapi import GoogleSearch  # type: ignore
    _HAS_SERPAPI = True
except Exception:
    _HAS_SERPAPI = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0 Safari/537.36"
}

def safe_ext_from_url(url: str) -> str:
    path = urlsplit(url).path
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext in ("jpg", "jpeg", "png", "gif", "webp", "bmp", "tiff", "tif"):
        return ext if ext != "jpeg" else "jpg"
    return "jpg"

def normalize_and_hash(img_path: Path, dest_dir: Path, min_size=(64,64)):
    """
    Abre imagen, valida tamaño, convierte a RGB, guarda como JPEG y devuelve metadata + phash.
    """
    meta = {"src": str(img_path), "ok": False, "reason": None}
    try:
        with Image.open(img_path) as im:
            # force RGB
            if im.mode != "RGB":
                im = im.convert("RGB")
            w,h = im.size
            if w < min_size[0] or h < min_size[1]:
                meta["reason"] = f"too_small ({w}x{h})"
                return meta
            # recompress to JPEG to normalize extension/format (lossy but okay for dataset)
            # compute hash on resized copy (phash)
            ph = imagehash.phash(im)
            meta["phash"] = str(ph)
            meta["width"] = w
            meta["height"] = h
            meta["ok"] = True
            # don't save here; caller will handle dedup and final move
            return meta
    except UnidentifiedImageError:
        meta["reason"] = "unidentified_image"
        return meta
    except Exception as e:
        meta["reason"] = f"error:{repr(e)}"
        return meta

def download_url_to(path: Path, url: str, timeout=10, max_bytes=None):
    """
    Descarga (requests) y guarda en path. Maneja reintentos simples.
    """
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout, stream=True)
            if r.status_code != 200:
                return False, f"status_{r.status_code}"
            # opcional: limitar tamaño
            with open(path, "wb") as f:
                total = 0
                for chunk in r.iter_content(1024*8):
                    if chunk:
                        f.write(chunk)
                        total += len(chunk)
                        if max_bytes and total > max_bytes:
                            return False, "too_big"
            return True, None
        except Exception as e:
            time.sleep(0.3 + attempt*0.5)
            continue
    return False, "download_failed"

def serpapi_images_for_query(query: str, out_dir: Path, max_num=300, serpapi_key=None):
    """
    Usa SerpAPI (paginado) para obtener 'max_num' URLs y descarga con requests.
    """
    if not _HAS_SERPAPI:
        raise RuntimeError("serpapi no instalado")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = {"engine": "google_images", "q": query, "api_key": serpapi_key, "ijn":"0"}
    count = 0
    page = 0
    while count < max_num:
        params["ijn"] = str(page)
        gs = GoogleSearch(params)
        res = gs.get_dict()
        imgs = res.get("images_results", [])
        if not imgs:
            break
        for i, item in enumerate(imgs):
            if count >= max_num:
                break
            url = item.get("original") or item.get("thumbnail") or item.get("src")
            if not url:
                continue
            ext = safe_ext_from_url(url)
            fname = out_dir / f"serp_{count}.{ext}"
            ok, reason = download_url_to(fname, url, timeout=10, max_bytes=10_000_000)
            if not ok:
                if fname.exists():
                    fname.unlink()
            else:
                count += 1
        page += 1
        if page > 50:
            break
    return

def icrawler_download(query: str, out_dir: Path, max_num=300, engine="google"):
    """
    Lanza icrawler; luego haremos post-process del contenido descargado.
    """
    if not _HAS_ICRAWLER:
        raise RuntimeError("icrawler no instalado")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_num_int = int(max_num)
    # icrawler Google tiene limitaciones: si max_num>1000 google devuelve warning. Aun así lo intento.
    if engine == "google":
        crawler = GoogleImageCrawler(storage={'root_dir': str(out_dir)}, feeder_threads=1, parser_threads=1, downloader_threads=4)
    else:
        crawler = BingImageCrawler(storage={'root_dir': str(out_dir)}, feeder_threads=1, parser_threads=1, downloader_threads=4)
    # mínimo size en icrawler evita iconos pequeños
    crawler.crawl(keyword=query, max_num=max_num_int, min_size=(100,100))

def postprocess_folder(raw_dir: Path, final_dir: Path, min_size=(64,64), dedup_hamming=6, keep_meta=True, face_filter=False, verbose=True):
    """
    Valida/normaliza y deduplica las imágenes descargadas.
    - raw_dir: directorio con archivos (icrawler/serpapi output)
    - final_dir: dónde copiar las imágenes válidas y únicas
    - dedup_hamming: hamming threshold para phash (<= es considerado duplicado)
    """
    raw_dir = Path(raw_dir)
    final_dir = Path(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in raw_dir.iterdir() if p.is_file()])
    meta_list = []
    hash_index = {}  # phash_str -> filename
    kept = 0

    for p in tqdm(files, desc=f"Postprocess {raw_dir.name}", disable=not verbose):
        try:
            with Image.open(p) as im:
                # normalize to RGB
                try:
                    if im.mode != "RGB":
                        im = im.convert("RGB")
                except Exception:
                    pass
                w,h = im.size
                if w < min_size[0] or h < min_size[1]:
                    meta_list.append({"src": str(p), "ok": False, "reason": f"too_small {w}x{h}"})
                    continue
                # optional face filter
                if face_filter and _HAS_FACE:
                    # convert to array for face_recognition
                    import numpy as np
                    arr = np.array(im)
                    faces = face_recognition.face_locations(arr, model="hog")
                    if len(faces) == 0:
                        meta_list.append({"src": str(p), "ok": False, "reason": "no_face"})
                        continue
                # compute hash
                ph = imagehash.phash(im)
                phs = str(ph)
                is_dup = False
                # check against existing hashes (fast Hamming)
                for existing_ph, fname in hash_index.items():
                    # compare hamming distance
                    hd = imagehash.hex_to_hash(phs) - imagehash.hex_to_hash(existing_ph)
                    if hd <= dedup_hamming:
                        is_dup = True
                        meta_list.append({"src": str(p), "ok": False, "reason": f"dupe_of:{fname}", "phash": phs})
                        break
                if is_dup:
                    continue
                # keep image: save normalized jpg in final_dir with sequential name
                out_name = f"img_{kept:06d}.jpg"
                out_path = final_dir / out_name
                im.save(out_path, format="JPEG", quality=92)
                hash_index[phs] = out_name
                meta_list.append({"src": str(p), "ok": True, "dest": str(out_path), "phash": phs, "width": w, "height": h})
                kept += 1
        except UnidentifiedImageError:
            meta_list.append({"src": str(p), "ok": False, "reason": "unidentified"})
        except Exception as e:
            meta_list.append({"src": str(p), "ok": False, "reason": f"error:{repr(e)}"})

    # save metadata summary
    meta_path = final_dir / "_download_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, indent=2, ensure_ascii=False)
    # save basic csv
    df = pd.DataFrame(meta_list)
    df.to_csv(final_dir / "_download_summary.csv", index=False)
    return final_dir, meta_path

def run_for_query(query: str, out_root: Path, tag: str, engine: str, num: int, serpapi_key=None, **kwargs):
    tmp_dir = out_root / f"{tag}_raw"
    final_dir = out_root / f"{tag}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    logging.info(f"Downloading {num} images for '{query}' into {tmp_dir} using {engine}")
    if engine == "serpapi":
        if not _HAS_SERPAPI:
            raise RuntimeError("SerpAPI not available. Install serpapi or use icrawler.")
        serpapi_images_for_query(query, tmp_dir, max_num=num, serpapi_key=serpapi_key)
    else:
        if not _HAS_ICRAWLER:
            raise RuntimeError("icrawler not available. Install icrawler or use serpapi.")
        # icrawler might warn and clamp google to 1000; we still call it
        icrawler_download(query, tmp_dir, max_num=num, engine=engine)
    # postprocess
    return postprocess_folder(tmp_dir, final_dir, **kwargs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pos', default="Linda Hamilton")
    parser.add_argument('--neg', default="middle age woman")
    parser.add_argument('--num', type=int, default=6000)
    parser.add_argument('--engine', choices=['google','bing','serpapi'], default='google')
    parser.add_argument('--out', default='data_linda/raw')
    parser.add_argument('--serpapi_key', default=d8767f2538f218d9eaeda7888e1a8864db2b63951a174716020cf4d35e8447a7)
    parser.add_argument('--min_size', type=int, nargs=2, default=(100,100), help="min width height")
    parser.add_argument('--dedup_hamming', type=int, default=6, help="hamming distance for phash dedup")
    parser.add_argument('--face_filter', default="true", action='store_true', help="keep only images with faces (requires face_recognition)")
    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    for query, tag in [(args.pos, "pos"), (args.neg, "neg")]:
        try:
            final_dir, meta = run_for_query(query, out_root, tag, engine=args.engine, num=args.num,
                                            serpapi_key=args.serpapi_key, min_size=tuple(args.min_size),
                                            dedup_hamming=args.dedup_hamming, face_filter=args.face_filter)
            logging.info(f"Finished {tag}: final_dir={final_dir}, meta={meta}")
        except Exception as e:
            logging.exception("Failed for %s: %s", tag, e)

if __name__ == "__main__":
    main()
