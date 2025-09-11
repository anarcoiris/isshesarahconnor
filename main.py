#!/usr/bin/env python3
"""
pipeline_all_in_one.py

Unifica: fetch (SerpAPI/icrawler), preprocess (crop faces, normalize, dedup), train (robust AMP/resume),
diagnose, export (TorchScript/ONNX) y serve (Flask) con una GUI para orquestarlo todo.

Uso:
    python pipeline_all_in_one.py --gui

Notas:
 - Muchas librerías son opcionales; el script detecta disponibilidad y usa fallbacks.
 - Si usas SerpAPI, pon tu API key en la GUI campo 'SerpAPI key' o usa icrawler.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import csv
import datetime
import io
import json
import logging
import os
import queue
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError, ImageTk
ImageFile.LOAD_TRUNCATED_IMAGES = True

# optional libs (graceful fallback)
try:
    import face_recognition  # type: ignore
    _HAS_FACE_REC = True
except Exception:
    _HAS_FACE_REC = False

try:
    from facenet_pytorch import MTCNN  # type: ignore
    _HAS_FACENET = True
except Exception:
    _HAS_FACENET = False

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

try:
    import imagehash  # type: ignore
    _HAS_IMGHASH = True
except Exception:
    _HAS_IMGHASH = False

# fetch helpers libs
try:
    from icrawler.builtin import GoogleImageCrawler, BingImageCrawler  # type: ignore
    _HAS_ICRAWLER = True
except Exception:
    _HAS_ICRAWLER = False

try:
    from serpapi import GoogleSearch  # type: ignore
    _HAS_SERPAPI = True
except Exception:
    _HAS_SERPAPI = False

# ML libs
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms
from torch.optim import AdamW

# GUI
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox, simpledialog
    _HAS_TK = True
except Exception:
    _HAS_TK = False

# web server
from flask import Flask, request, jsonify, render_template_string

# data & utils
import pandas as pd
from urllib.parse import urlsplit
import requests

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("pipeline_all_in_one")

# ---------------------------
# Constants & defaults
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent
BLACKLIST_FILE = BASE_DIR / "blacklist_domains.txt"

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

HEADERS = {"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0 Safari/537.36"}

# ---------------------------
# Utilities
# ---------------------------
def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_json(p: Path, obj):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def load_json(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def sanitize_foldername(s: str, maxlen: int = 60) -> str:
    s = s.strip()
    s = re.sub(r'\s+', '_', s)
    s = re.sub(r'[^\w\-_]', '', s)
    return s[:maxlen] or "query"

# ---------------------------
# Downloader: serpapi + requests with blacklist logging
# ---------------------------
def safe_ext_from_url(url: str) -> str:
    path = urlsplit(url).path
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext in ("jpg","jpeg","png","gif","webp","bmp","tiff","tif"):
        return "jpg" if ext=="jpeg" else ext
    return "jpg"

def download_url_to(path: Path, url: str, timeout=10, max_bytes=None) -> Tuple[bool, Optional[str]]:
    domain = urlsplit(url).netloc
    # check blacklist
    if BLACKLIST_FILE.exists():
        try:
            bl = [l.strip() for l in open(BLACKLIST_FILE, "r", encoding="utf-8").read().splitlines() if l.strip()]
            for d in bl:
                if d and d in domain:
                    return False, "blacklisted_domain"
        except Exception:
            pass
    for attempt in range(3):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout, stream=True, verify=True)
            if r.status_code == 403:
                # log domain
                try:
                    with open(BLACKLIST_FILE, "a", encoding="utf-8") as f:
                        f.write(domain + "\n")
                except Exception:
                    pass
                return False, "status_403"
            if r.status_code != 200:
                return False, f"status_{r.status_code}"
            with open(path, "wb") as f:
                total = 0
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)
                        total += len(chunk)
                        if max_bytes and total > max_bytes:
                            return False, "too_big"
            return True, None
        except requests.exceptions.SSLError:
            try:
                with open(BLACKLIST_FILE, "a", encoding="utf-8") as f:
                    f.write(domain + "\n")
            except Exception:
                pass
            return False, "ssl_error"
        except Exception:
            time.sleep(0.5 + attempt*0.5)
            continue
    return False, "download_failed"

def serpapi_images_for_query(query: str, out_dir: Path, max_num=300, serpapi_key: Optional[str]=None):
    out_dir = Path(out_dir); safe_mkdir(out_dir)
    if not _HAS_SERPAPI or not serpapi_key:
        raise RuntimeError("SerpAPI no disponible o serpapi_key no proporcionada")
    params = {"engine":"google_images","q":query,"api_key":serpapi_key,"ijn":"0"}
    count = 0; page = 0
    while count < max_num:
        params["ijn"] = str(page)
        gs = GoogleSearch(params)
        res = gs.get_dict()
        imgs = res.get("images_results", []) or []
        if not imgs:
            break
        for item in imgs:
            if count >= max_num:
                break
            url = item.get("original") or item.get("thumbnail") or item.get("src")
            if not url:
                continue
            ext = safe_ext_from_url(url)
            fname = out_dir / f"img_{count}.{ext}"
            ok, reason = download_url_to(fname, url, timeout=12, max_bytes=10_000_000)
            if not ok:
                if fname.exists(): fname.unlink()
            else:
                count += 1
        page += 1
        if page > 50:
            break

def icrawler_download(query: str, out_dir: Path, max_num=300, engine="google"):
    if not _HAS_ICRAWLER:
        raise RuntimeError("icrawler no instalado")
    out_dir = Path(out_dir); safe_mkdir(out_dir)
    max_num_int = int(max_num)
    if engine == "google":
        crawler = GoogleImageCrawler(storage={'root_dir': str(out_dir)}, feeder_threads=1, parser_threads=1, downloader_threads=4)
    else:
        crawler = BingImageCrawler(storage={'root_dir': str(out_dir)}, feeder_threads=1, parser_threads=1, downloader_threads=4)
    crawler.crawl(keyword=query, max_num=max_num_int, min_size=(100,100))

# ---------------------------
# Face detection & crop utilities (tries multiple backends)
# ---------------------------
def detect_faces_pil(img: Image.Image) -> List[Tuple[int,int,int,int]]:
    # return list of boxes (left, top, right, bottom)
    if _HAS_FACE_REC:
        try:
            arr = np.array(img.convert("RGB"))
            boxes = face_recognition.face_locations(arr, model="hog")
            return [(left, top, right, bottom) for (top, right, bottom, left) in boxes]
        except Exception:
            pass
    if _HAS_FACENET:
        try:
            mtcnn = MTCNN(keep_all=True, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            arr = np.array(img.convert("RGB"))
            boxes, _ = mtcnn.detect(arr)
            if boxes is None:
                return []
            out = []
            for box in boxes:
                l, t, r, b = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                out.append((l, t, r, b))
            return out
        except Exception:
            pass
    if _HAS_CV2:
        try:
            arr = np.array(img.convert("RGB"))
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            detector = cv2.CascadeClassifier(cascade_path)
            rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(24,24))
            boxes = []
            for (x,y,w,h) in rects:
                boxes.append((int(x),int(y),int(x+w),int(y+h)))
            return boxes
        except Exception:
            pass
    return []

def crop_faces_from_image(src_path: Path, dest_dir: Path, margin: float=0.2, min_size: int=48, keep_no_face=False) -> List[Path]:
    saved = []
    try:
        with Image.open(src_path) as im:
            im = im.convert("RGB")
            w,h = im.size
            boxes = detect_faces_pil(im)
            if not boxes and keep_no_face:
                target = dest_dir / "noface"; safe_mkdir(target)
                outf = target / src_path.name
                try:
                    im.save(outf, format="JPEG", quality=92)
                    saved.append(outf)
                except Exception:
                    pass
                return saved
            for i,(l,t,r,b) in enumerate(boxes):
                fw = r-l; fh = b-t
                if fw < min_size or fh < min_size:
                    continue
                pad_w = int(fw * margin); pad_h = int(fh * margin)
                L = max(0, l-pad_w); T = max(0, t-pad_h); R = min(w, r+pad_w); B = min(h, b+pad_h)
                crop = im.crop((L,T,R,B))
                out_dir = Path(dest_dir); safe_mkdir(out_dir)
                out_name = out_dir / f"{src_path.stem}_face_{i}.jpg"
                crop.save(out_name, format="JPEG", quality=92)
                saved.append(out_name)
    except UnidentifiedImageError:
        pass
    except Exception:
        logger.exception("Error cropping %s", src_path)
    return saved

def crop_faces_in_folder(raw_dir: Path, out_dir: Path, margin=0.2, min_size=48, keep_no_face=False):
    raw_dir = Path(raw_dir); out_dir = Path(out_dir)
    safe_mkdir(out_dir)
    stats = {"images":0, "faces":0, "skipped_no_face":0, "errors":0}
    imgs = [p for p in raw_dir.rglob("*") if p.suffix.lower() in (".jpg",".jpeg",".png",".webp",".bmp")]
    for p in imgs:
        stats["images"] += 1
        saved = crop_faces_from_image(p, out_dir, margin=margin, min_size=min_size, keep_no_face=keep_no_face)
        if not saved:
            stats["skipped_no_face"] += 1
        else:
            stats["faces"] += len(saved)
    return stats

def generate_metadata_for_crops(raw_dir: Path, crop_dir: Path, out_meta: Path):
    rows = []
    for p in sorted(Path(crop_dir).rglob("*.jpg")):
        base = p.name.split("_face_")[0]
        candidates = list(Path(raw_dir).rglob(f"{base}*"))
        src = str(candidates[0]) if candidates else ""
        rows.append({"crop": str(p), "src": src, "timestamp": datetime.datetime.utcnow().isoformat()})
    save_json(out_meta.with_suffix(".json"), rows)
    try:
        df = pd.DataFrame(rows)
        df.to_csv(out_meta.with_suffix(".csv"), index=False)
    except Exception:
        logger.exception("Failed saving metadata csv")

# ---------------------------
# Prepare dataset (internal compact implementation)
# ---------------------------
def file_hash(path: Path, chunk_size: int = 8192) -> str:
    h = None
    try:
        import hashlib
        h = hashlib.md5()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""

def ensure_unique_filename_internal(dest_dir: Path, base_name: str, ext: str, max_tries: int = 1000) -> Path:
    candidate = dest_dir / f"{base_name}.{ext}"
    if not candidate.exists():
        return candidate
    for i in range(1, max_tries):
        candidate = dest_dir / f"{base_name}_{i}.{ext}"
        if not candidate.exists():
            return candidate
    import time, hashlib
    suffix = hashlib.md5((base_name + str(time.time())).encode()).hexdigest()[:8]
    return dest_dir / f"{base_name}_{suffix}.{ext}"

def internal_prepare_dataset(raw_root: Path, out_root: Path, train_ratio: float, val_ratio: float, test_ratio: float,
                             min_size=(50,50), dedupe_phash: bool=False, phash_threshold: int=6, seed: int=42):
    """
    Compact internal implementation:
     - canonicalizes labels by folder name
     - keeps unique images by MD5
     - optional perceptual dedupe if imagehash installed
     - splits and copies to out_root/train|val|test/<label>
    """
    raw_root = Path(raw_root)
    out_root = Path(out_root)
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root {raw_root} not found")
    # detect top-level label dirs
    labels_raw = [p.name for p in raw_root.iterdir() if p.is_dir()]
    def canonical_label(name: str) -> str:
        n = name.strip().lower()
        for suf in ['_raw','-raw','_images','-images','_img']:
            if n.endswith(suf):
                n = n[:-len(suf)]
        return n
    labels_by_canon = {}
    for lbl in labels_raw:
        canon = canonical_label(lbl)
        labels_by_canon.setdefault(canon, []).append(lbl)
    logger.info("Labels (canonical -> raw): %s", labels_by_canon)
    items_by_label = {c: [] for c in labels_by_canon}
    seen_md5 = {}
    total_examined = 0
    total_valid = 0
    for canon, raws in labels_by_canon.items():
        for rd in raws:
            pdir = raw_root / rd
            for p in pdir.rglob("*"):
                if not p.is_file():
                    continue
                total_examined += 1
                # try open and validate
                try:
                    with Image.open(p) as im:
                        im.load()
                        if im.mode == 'P':
                            im = im.convert('RGBA')
                        if im.mode == 'LA':
                            im = im.convert('RGBA')
                        if im.mode == 'CMYK':
                            im = im.convert('RGB')
                        if im.mode == 'RGBA':
                            bg = Image.new("RGB", im.size, (255,255,255))
                            bg.paste(im, mask=im.split()[3])
                            im = bg
                        else:
                            im = im.convert('RGB')
                        w,h = im.size
                        if w < min_size[0] or h < min_size[1]:
                            continue
                except Exception:
                    continue
                md5 = file_hash(p)
                if not md5:
                    continue
                if md5 in seen_md5:
                    continue
                seen_md5[md5] = str(p)
                items_by_label[canon].append((p, md5))
                total_valid += 1
    logger.info("Examined %d files, found %d valid unique images", total_examined, total_valid)
    # optional phash dedupe
    if dedupe_phash and _HAS_IMGHASH:
        import imagehash
        phash_map = {}
        kept = {c: [] for c in items_by_label}
        for canon, items in items_by_label.items():
            for p, md5 in items:
                try:
                    with Image.open(p) as im:
                        im = im.convert("RGB")
                        ph = imagehash.phash(im)
                except Exception:
                    kept[canon].append((p, md5))
                    continue
                collided = False
                for eph in phash_map.keys():
                    if (ph - eph) <= phash_threshold:
                        collided = True
                        break
                if not collided:
                    phash_map[ph] = (p, md5)
                    kept[canon].append((p, md5))
        items_by_label = kept
        total_valid = sum(len(v) for v in items_by_label.values())
        logger.info("After phash dedupe: %d images remain", total_valid)
    elif dedupe_phash:
        logger.warning("imagehash not installed; skipping phash dedupe")
    # split and copy
    import random
    random.seed(seed)
    out_root.mkdir(parents=True, exist_ok=True)
    summary = {}
    for canon, items in items_by_label.items():
        n = len(items)
        if n == 0:
            logger.warning("No images for label '%s', skipping", canon)
            continue
        random.shuffle(items)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        if n_train == 0 and n >= 1:
            n_train = 1
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        train = items[:n_train]
        val = items[n_train:n_train + n_val]
        test = items[n_train + n_val:]
        summary[canon] = {'total': n, 'train': len(train), 'val': len(val), 'test': len(test)}
        for split_name, split_list in [('train', train), ('val', val), ('test', test)]:
            target_dir = out_root / split_name / canon
            target_dir.mkdir(parents=True, exist_ok=True)
            for src_path, md5 in split_list:
                ext = src_path.suffix.lower().lstrip('.') or 'jpg'
                base_name = src_path.stem
                safe_base = "".join(c for c in base_name if c.isalnum() or c in ('-','_')).strip()[:60] or md5[:8]
                dest_path = ensure_unique_filename_internal(target_dir, safe_base, ext)
                try:
                    if dest_path.exists():
                        if file_hash(dest_path) == md5:
                            continue
                    shutil.copy2(src_path, dest_path)
                except Exception:
                    try:
                        im = Image.open(src_path)
                        im = im.convert('RGB')
                        fallback_name = ensure_unique_filename_internal(target_dir, safe_base, 'jpg')
                        im.save(fallback_name, format='JPEG', quality=92)
                    except Exception:
                        continue
    logger.info("Dataset prepared at %s", out_root)
    logger.info("Summary: %s", summary)
    return summary

# ---------------------------
# Training utilities (compact)
# ---------------------------
def seed_everything(seed: int=42):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def make_transforms(image_size: int = 224):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(int(image_size*1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return train_transform, val_transform

def build_model(num_classes:int, base_model:str='resnet50', pretrained:bool=True):
    if base_model=='resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    elif base_model=='resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError("base_model no soportado")
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

def get_parameter_groups(model: nn.Module, weight_decay: float=1e-4):
    decay=[]; no_decay=[]
    for n,p in model.named_parameters():
        if not p.requires_grad: continue
        if len(p.shape)==1 or n.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params":decay,"weight_decay":weight_decay},{"params":no_decay,"weight_decay":0.0}]

class DummyGradScaler:
    def scale(self, loss): return loss
    def step(self, opt): opt.step()
    def update(self): return None
    def unscale_(self, opt): return None

def create_grad_scaler(use_amp: bool):
    if not use_amp:
        return DummyGradScaler()
    try:
        from torch.amp import GradScaler
        return GradScaler()
    except Exception:
        return DummyGradScaler()

def autocast_ctx(enabled: bool):
    try:
        from torch.amp import autocast
        return autocast(enabled=enabled)
    except TypeError:
        try:
            if enabled:
                return autocast()
            return contextlib.nullcontext()
        except Exception:
            return contextlib.nullcontext()

def create_sampler_if_needed(dataset: datasets.ImageFolder):
    targets = [s[1] for s in dataset.samples]
    classes = sorted(set(targets))
    class_sample_count = np.array([targets.count(t) for t in classes]) if len(targets) else np.array([])
    if class_sample_count.size and (class_sample_count.max()/(class_sample_count.min()+1e-12) > 1.5):
        weights = 1.0 / torch.tensor([class_sample_count[t] for t in targets], dtype=torch.double)
        return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    return None

def train_one_epoch(model, loader, criterion, optimizer, device, scaler, stop_event, max_grad_norm=None):
    model.train()
    running_loss=0.0; n_samples=0
    for inputs, labels in loader:
        if stop_event.is_set():
            logger.info("Stop requested - breaking epoch")
            break
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast_ctx(isinstance(scaler, object) and not isinstance(scaler, DummyGradScaler)):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaled = scaler.scale(loss)
        scaled.backward()
        if max_grad_norm is not None:
            try:
                if hasattr(scaler, "unscale_"):
                    scaler.unscale_(optimizer)
            except Exception:
                pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        try:
            scaler.step(optimizer)
            scaler.update()
        except Exception:
            optimizer.step()
        bs = inputs.size(0)
        running_loss += float(loss.item()) * bs
        n_samples += bs
    return running_loss / max(1, n_samples)

def evaluate(model, loader, criterion, device):
    model.eval()
    val_loss=0.0; y_true=[]; y_pred=[]; n=0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += float(loss.item()) * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            n += inputs.size(0)
    acc = float(np.mean(np.array(y_true) == np.array(y_pred))) if len(y_true) else 0.0
    return {"val_loss": val_loss / max(1, n), "accuracy": acc, "y_true": y_true, "y_pred": y_pred}

def run_training(cfg: Dict, gui_queue: Optional[queue.Queue]=None, stop_event: Optional[threading.Event]=None):
    seed_everything(cfg.get("seed",42))
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("use_cuda",True) else "cpu")
    logger.info("Training device: %s", device)
    models_dir = Path(cfg.get("models_dir","models")); safe_mkdir(models_dir)
    train_tf, val_tf = make_transforms(cfg.get("image_size",224))
    train_dir = Path(cfg["data"]) / "train"
    val_dir = Path(cfg["data"]) / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("train/val folders missing under --data")
    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)
    logger.info("Classes: %s", train_ds.classes)
    sampler = None
    if cfg.get("use_sampler", True):
        sampler = create_sampler_if_needed(train_ds)
        if sampler is not None: logger.info("Using WeightedRandomSampler")
    train_loader = DataLoader(train_ds, batch_size=cfg.get("batch",32), shuffle=(sampler is None), sampler=sampler, num_workers=cfg.get("num_workers",4), pin_memory=(device.type=="cuda"))
    val_loader = DataLoader(val_ds, batch_size=cfg.get("batch",32), shuffle=False, num_workers=cfg.get("num_workers",4), pin_memory=(device.type=="cuda"))
    model = build_model(cfg.get("num_classes", len(train_ds.classes)), base_model=cfg.get("base_model","resnet50"), pretrained=cfg.get("pretrained", True)).to(device)
    freeze_epochs = cfg.get("freeze_epochs",3)
    if freeze_epochs > 0:
        for n,p in model.named_parameters():
            if "fc" not in n:
                p.requires_grad = False
    criterion = nn.CrossEntropyLoss()
    opt = AdamW(get_parameter_groups(model, weight_decay=cfg.get("weight_decay",1e-4)), lr=cfg.get("lr",1e-4))
    if cfg.get("scheduler","reduce_on_plateau")=="reduce_on_plateau":
        try:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3, verbose=True)
        except Exception:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=3)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1,cfg.get("epochs",20)))
    use_amp = cfg.get("amp", False) and device.type=="cuda"
    scaler = create_grad_scaler(use_amp)
    start_epoch=1; best_val=1e9; history=[]
    if cfg.get("resume"):
        try:
            ckpt = torch.load(cfg["resume"], map_location=device)
            state = ckpt.get("model_state", ckpt)
            new_state = {}
            for k,v in state.items():
                nk = k.replace("module.","")
                new_state[nk] = v
            model.load_state_dict(new_state)
            try:
                opt.load_state_dict(ckpt["optimizer_state"])
                start_epoch = ckpt.get("epoch",1)+1
                best_val = ckpt.get("best_val_loss",best_val)
            except Exception:
                logger.warning("Could not load optimizer state")
            logger.info("Resumed from %s", cfg["resume"])
        except Exception:
            logger.exception("Failed to resume")
    stop_event = stop_event or threading.Event()
    epochs_no_improve = 0
    for epoch in range(start_epoch, cfg.get("epochs",20)+1):
        t0=time.time()
        if epoch == freeze_epochs + 1:
            logger.info("Unfreezing backbone")
            for p in model.parameters(): p.requires_grad=True
            opt = AdamW(get_parameter_groups(model, weight_decay=cfg.get("weight_decay",1e-4)), lr=cfg.get("lr",1e-4)*cfg.get("ft_lr_scale",0.1))
        train_loss = train_one_epoch(model, train_loader, criterion, opt, device, scaler, stop_event, max_grad_norm=cfg.get("grad_clip",None))
        eval_res = evaluate(model, val_loader, criterion, device)
        val_loss = eval_res["val_loss"]; val_acc = eval_res["accuracy"]
        if cfg.get("scheduler","reduce_on_plateau")=="reduce_on_plateau":
            try: scheduler.step(val_loss)
            except Exception: pass
        else:
            try: scheduler.step()
            except Exception: pass
        elapsed = time.time()-t0
        logger.info("Epoch %d/%d train_loss=%.6f val_loss=%.6f val_acc=%.4f time=%.1fs", epoch, cfg.get("epochs",20), train_loss, val_loss, val_acc, elapsed)
        ckpt_path = Path(models_dir)/f"ckpt_epoch_{epoch}.pth"
        payload = {"epoch":epoch, "model_state":model.state_dict(), "best_val_loss":best_val}
        try:
            payload["optimizer_state"] = opt.state_dict()
        except Exception:
            pass
        try:
            torch.save(payload, ckpt_path)
        except Exception:
            logger.exception("Failed saving ckpt")
        if val_loss < best_val:
            best_val = val_loss
            try:
                torch.save(model.state_dict(), Path(models_dir)/"best_model.pth")
                logger.info("Saved best model")
            except Exception:
                logger.warning("Could not save best model")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        history.append({"epoch":epoch,"train_loss":train_loss,"val_loss":val_loss,"val_acc":val_acc,"time":elapsed})
        if gui_queue is not None:
            gui_queue.put({"type":"epoch","epoch":epoch,"train_loss":train_loss,"val_loss":val_loss,"val_acc":val_acc})
        if cfg.get("early_stopping_patience",5) > 0 and epochs_no_improve >= cfg.get("early_stopping_patience",5):
            logger.info("Early stopping")
            break
        if stop_event.is_set():
            logger.info("Stop requested by GUI")
            break
    try:
        pd.DataFrame(history).to_csv(Path(models_dir)/"training_history.csv", index=False)
    except Exception:
        pass
    logger.info("Training finished. Best val: %.6f", best_val)
    if gui_queue is not None:
        gui_queue.put({"type":"done","best_val":best_val})

# ---------------------------
# Export utilities
# ---------------------------
def export_model_torchscript(model_state_path: str, out_path: str, base_model: str='resnet50', num_classes:int=2):
    model = build_model(num_classes=num_classes, base_model=base_model, pretrained=False)
    model.load_state_dict(torch.load(model_state_path, map_location="cpu"))
    model.eval()
    example = torch.randn(1,3,224,224)
    traced = torch.jit.trace(model, example)
    traced.save(out_path)
    return out_path

def export_model_onnx(model_state_path: str, out_path: str, base_model: str='resnet50', num_classes:int=2):
    model = build_model(num_classes=num_classes, base_model=base_model, pretrained=False)
    model.load_state_dict(torch.load(model_state_path, map_location="cpu"))
    model.eval()
    dummy = torch.randn(1,3,224,224)
    torch.onnx.export(model, dummy, out_path, input_names=['input'], output_names=['output'], opset_version=12)
    return out_path

# ---------------------------
# Flask serve (in-process, runs in background thread)
# ---------------------------
FLASK_TEMPLATE = """
<!doctype html>
<title>Classifier</title>
<h2>Upload image</h2>
<form method=post enctype=multipart/form-data action="/predict">
  <input type=file name=image>
  <input type=submit value=Upload>
</form>
{% if prediction %}
  <h3>Prediction: {{prediction}} (p={{prob:.4f}})</h3>
  <img src="data:image/png;base64,{{preview_b64}}" style="max-width:320px"/>
{% endif %}
"""
def create_flask_app(model_state_path: Optional[str], labels: Optional[List[str]]=None, base_model:str='resnet50'):
    app = Flask("pipeline_inference")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Flask inference device: %s", device)
    num_classes = len(labels) if labels else 2
    model = build_model(num_classes=num_classes, base_model=base_model, pretrained=False)
    if model_state_path and Path(model_state_path).exists():
        try:
            sd = torch.load(model_state_path, map_location="cpu")
            if isinstance(sd, dict) and "model_state" in sd:
                sd = sd["model_state"]
            new = {k.replace("module.",""):v for k,v in sd.items()}
            model.load_state_dict(new)
            logger.info("Loaded inference model from %s", model_state_path)
        except Exception:
            logger.exception("Failed loading model for inference")
    model = model.to(device).eval()
    transform_local = DEFAULT_TRANSFORM

    @app.route("/", methods=["GET"])
    def index():
        return render_template_string(FLASK_TEMPLATE)

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            if "image" not in request.files:
                return render_template_string(FLASK_TEMPLATE, prediction=None, error="No file")
            b = request.files["image"].read()
            pil = Image.open(io.BytesIO(b))
            if pil.mode != "RGB": pil = pil.convert("RGB")
            t = transform_local(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(t)
                probs = F.softmax(logits, dim=-1).cpu().numpy().ravel().tolist()
                idx = int(np.argmax(probs))
            pred = labels[idx] if labels else str(idx)
            pb = probs[idx]
            buf = io.BytesIO(); pil.thumbnail((320,320)); pil.save(buf, format="PNG")
            preview_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            return render_template_string(FLASK_TEMPLATE, prediction=pred, prob=pb, preview_b64=preview_b64)
        except Exception:
            logger.exception("Predict error")
            return jsonify({"error": "server error"}), 500

    @app.route("/predict_json", methods=["POST"])
    def predict_json():
        try:
            if "image" in request.files:
                b = request.files["image"].read()
            else:
                data = request.get_json(silent=True) or {}
                b64 = data.get("image_b64")
                if not b64:
                    return jsonify({"error":"no image"}), 400
                b = base64.b64decode(b64)
            pil = Image.open(io.BytesIO(b)); pil = pil.convert("RGB")
            t = transform_local(pil).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(t)
                probs = F.softmax(logits, dim=-1).cpu().numpy().ravel().tolist()
                idx = int(np.argmax(probs))
            return jsonify({"pred_index": idx, "pred_label": labels[idx] if labels else str(idx), "probabilities": {labels[i] if labels else str(i): float(probs[i]) for i in range(len(probs))}})
        except Exception:
            logger.exception("predict_json error")
            return jsonify({"error":"server error"}), 500

    return app

# ---------------------------
# Labeler support classes
# ---------------------------
def make_thumbnail(path_or_pil, size=(340,340)) -> Optional[Image.Image]:
    try:
        if isinstance(path_or_pil, (str, Path)):
            im = Image.open(path_or_pil)
        else:
            im = path_or_pil
        im = im.convert("RGB")
        if hasattr(Image, "Resampling"):
            im.thumbnail(size, Image.Resampling.LANCZOS)
        else:
            im.thumbnail(size, Image.ANTIALIAS)
        thumb = Image.new("RGB", size, (40,40,40))
        x = (size[0] - im.width) // 2
        y = (size[1] - im.height) // 2
        thumb.paste(im, (x, y))
        return thumb
    except Exception:
        return None

class SchemaManager:
    def __init__(self):
        self.schema: Dict[str, Dict] = {
            "hair_color": {"type": "single", "labels": ["black", "brown", "blonde", "red", "gray", "other"], "default": "brown"},
            "eye_color":  {"type": "single", "labels": ["brown", "blue", "green", "other"], "default": "brown"},
            "accessories": {"type": "multi", "labels": ["glasses", "hat", "earrings"], "default": []}
        }
    def add_attr(self, name:str, attr_type:str, labels:List[str], default=None):
        self.schema[name] = {"type": attr_type, "labels": labels, "default": default if default is not None else (labels[0] if labels else "")}
    def remove_attr(self, name:str):
        if name in self.schema:
            del self.schema[name]

class AnnotationStore:
    def __init__(self, path: Optional[Path] = None):
        self.path: Optional[Path] = path
        self.data: Dict[str, Dict] = {}
    def load(self, path: Path):
        self.path = Path(path)
        if not self.path.exists():
            self.data = {}
            return
        with open(self.path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
    def save(self, path: Optional[Path] = None):
        p = Path(path) if path is not None else self.path
        if p is None:
            raise RuntimeError("No path provided for saving annotations")
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        self.path = p
    def set_annotation(self, img_path: str, ann: Dict):
        self.data[str(img_path)] = ann
    def get_annotation(self, img_path: str) -> Dict:
        return self.data.get(str(img_path), {})
    def bulk_apply(self, image_paths: List[str], attr: str, value, schema_mgr: SchemaManager):
        for ip in image_paths:
            cur = self.get_annotation(ip) or {}
            meta = schema_mgr.schema.get(attr)
            if meta is None:
                continue
            if meta["type"] == "single":
                cur[attr] = value
            else:
                lst = cur.get(attr, [])
                if value in lst:
                    lst.remove(value)
                else:
                    lst.append(value)
                cur[attr] = lst
            self.set_annotation(ip, cur)

class SchemaDialog:
    def __init__(self, parent, schema_mgr: SchemaManager):
        self.top = tk.Toplevel(parent)
        self.top.title("Schema Manager")
        self.schema_mgr = schema_mgr
        self.modified = False
        self.listbox = tk.Listbox(self.top, height=8)
        self.listbox.grid(column=0, row=0, columnspan=3, sticky="nsew")
        self._rebuild_list()
        ttk.Button(self.top, text="Add attr", command=self._add_attr).grid(column=0, row=1)
        ttk.Button(self.top, text="Remove selected", command=self._remove_selected).grid(column=1, row=1)
        ttk.Button(self.top, text="Close", command=self._close).grid(column=2, row=1)
    def _rebuild_list(self):
        self.listbox.delete(0, "end")
        for k, v in self.schema_mgr.schema.items():
            self.listbox.insert("end", f"{k} ({v['type']}): {','.join(v['labels'])}")
    def _add_attr(self):
        name = simpledialog.askstring("Name", "Attribute name", parent=self.top)
        if not name:
            return
        typ = simpledialog.askstring("Type", "Type: single or multi", parent=self.top, initialvalue="single")
        if typ not in ("single", "multi"):
            messagebox.showerror("Invalid", "Type must be 'single' or 'multi'")
            return
        labs = simpledialog.askstring("Labels", "Comma separated labels", parent=self.top, initialvalue="a,b")
        labels = [l.strip() for l in labs.split(",") if l.strip()] if labs else []
        self.schema_mgr.add_attr(name, typ, labels, default=(labels[0] if labels else None))
        self.modified = True
        self._rebuild_list()
    def _remove_selected(self):
        sel = self.listbox.curselection()
        if not sel: return
        idx = sel[0]
        key = list(self.schema_mgr.schema.keys())[idx]
        if messagebox.askyesno("Remove", f"Remove attribute {key}?"):
            self.schema_mgr.remove_attr(key)
            self.modified = True
            self._rebuild_list()
    def _close(self):
        self.top.destroy()

# ---------------------------
# GUI: orchestrator
# ---------------------------
class PipelineGUI:
    def __init__(self):
        if not _HAS_TK:
            raise RuntimeError("Tkinter not available")
        self.root = tk.Tk()
        self.root.title("Unified Pipeline GUI")
        frm = ttk.Frame(self.root, padding=8)
        frm.grid(sticky="nsew")
        nb = ttk.Notebook(frm)
        nb.grid(column=0,row=0, sticky="nsew")
        self.tab_fetch = ttk.Frame(nb); self.tab_prep = ttk.Frame(nb); self.tab_label = ttk.Frame(nb)
        self.tab_train = ttk.Frame(nb); self.tab_diag = ttk.Frame(nb); self.tab_export = ttk.Frame(nb); self.tab_serve = ttk.Frame(nb)
        nb.add(self.tab_fetch, text="Fetch"); nb.add(self.tab_prep, text="Prep"); nb.add(self.tab_label, text="Labeler")
        nb.add(self.tab_train, text="Train"); nb.add(self.tab_diag, text="Diagnose"); nb.add(self.tab_export, text="Export"); nb.add(self.tab_serve, text="Serve")

        # --- FETCH tab ---
        ttk.Label(self.tab_fetch, text="Positive query").grid(column=0,row=0,sticky="w")
        self.fetch_pos = ttk.Entry(self.tab_fetch, width=50); self.fetch_pos.insert(0,"Linda Hamilton"); self.fetch_pos.grid(column=1,row=0)
        ttk.Label(self.tab_fetch, text="Additional queries (one per line)").grid(column=0,row=1,sticky="nw")
        self.fetch_extra_text = tk.Text(self.tab_fetch, width=50, height=6)
        self.fetch_extra_text.insert("1.0", "")
        self.fetch_extra_text.grid(column=1,row=1,sticky="w")
        ttk.Label(self.tab_fetch, text="Num images per query").grid(column=0,row=2,sticky="w")
        self.fetch_num = ttk.Entry(self.tab_fetch, width=10); self.fetch_num.insert(0,"500"); self.fetch_num.grid(column=1,row=2,sticky="w")
        ttk.Label(self.tab_fetch, text="Out folder").grid(column=0,row=3,sticky="w")
        self.fetch_out = ttk.Entry(self.tab_fetch, width=60); self.fetch_out.insert(0,str(BASE_DIR/"data_fetch")); self.fetch_out.grid(column=1,row=3)
        ttk.Label(self.tab_fetch, text="SerpAPI key (optional)").grid(column=0,row=4,sticky="w")
        self.fetch_serp = ttk.Entry(self.tab_fetch, width=60); self.fetch_serp.grid(column=1,row=4)
        ttk.Button(self.tab_fetch, text="Run fetch", command=self.run_fetch_thread).grid(column=1,row=5,sticky="w")

        # --- PREP tab ---
        ttk.Label(self.tab_prep, text="Raw folder (input)").grid(column=0,row=0,sticky="w")
        self.raw_folder = ttk.Entry(self.tab_prep, width=70); self.raw_folder.insert(0,str(BASE_DIR/"data_fetch")); self.raw_folder.grid(column=1,row=0)
        ttk.Button(self.tab_prep, text="Browse", command=lambda: self.browse_entry(self.raw_folder)).grid(column=2,row=0)
        ttk.Label(self.tab_prep, text="Out folder (data/final)").grid(column=0,row=1,sticky="w")
        self.prep_out = ttk.Entry(self.tab_prep, width=70); self.prep_out.insert(0,str(BASE_DIR/"data_final")); self.prep_out.grid(column=1,row=1)
        ttk.Button(self.tab_prep, text="Browse", command=lambda: self.browse_entry(self.prep_out)).grid(column=2,row=1)
        ttk.Label(self.tab_prep, text="Train/Val/Test ratios").grid(column=0,row=2,sticky="w")
        ratio_frame = ttk.Frame(self.tab_prep); ratio_frame.grid(column=1,row=2,sticky="w")
        ttk.Label(ratio_frame, text="train").grid(column=0,row=0); self.r_train = ttk.Entry(ratio_frame, width=6); self.r_train.insert(0,"0.7"); self.r_train.grid(column=1,row=0)
        ttk.Label(ratio_frame, text="val").grid(column=2,row=0);   self.r_val = ttk.Entry(ratio_frame, width=6); self.r_val.insert(0,"0.15"); self.r_val.grid(column=3,row=0)
        ttk.Label(ratio_frame, text="test").grid(column=4,row=0);  self.r_test = ttk.Entry(ratio_frame, width=6); self.r_test.insert(0,"0.15"); self.r_test.grid(column=5,row=0)
        self.phash_var = tk.BooleanVar(value=False); ttk.Checkbutton(self.tab_prep, text="Dedupe perceptual (imagehash)", variable=self.phash_var).grid(column=1,row=3,sticky="w")
        ttk.Label(self.tab_prep, text="phash threshold (lower stricter)").grid(column=0,row=4,sticky="w")
        self.phash_thresh = ttk.Entry(self.tab_prep, width=6); self.phash_thresh.insert(0,"6"); self.phash_thresh.grid(column=1,row=4,sticky="w")
        ttk.Button(self.tab_prep, text="Prepare dataset (use scripts/prepare_dataset.py if present)", command=self.run_prepare_thread).grid(column=1,row=5,sticky="w")
        ttk.Label(self.tab_prep, text="Faces out (for labeler)").grid(column=0,row=6,sticky="w")
        self.faces_out = ttk.Entry(self.tab_prep, width=70); self.faces_out.insert(0,str(BASE_DIR/"data_faces")); self.faces_out.grid(column=1,row=6)
        ttk.Button(self.tab_prep, text="Crop faces", command=self.run_crop_thread).grid(column=1,row=7,sticky="w")
        ttk.Button(self.tab_prep, text="Generate metadata CSV/JSON", command=self.run_meta_thread).grid(column=1,row=8,sticky="w")

        # Labeler tab
        self.schema_mgr = SchemaManager()
        self.ann_store = AnnotationStore()
        ttk.Label(self.tab_label, text="Faces folder").grid(column=0,row=0,sticky="w")
        self.label_folder_e = ttk.Entry(self.tab_label, width=60); self.label_folder_e.insert(0,str(BASE_DIR/"data_faces")); self.label_folder_e.grid(column=1,row=0)
        ttk.Button(self.tab_label, text="Browse", command=lambda: self.browse_entry(self.label_folder_e)).grid(column=2,row=0)
        ttk.Button(self.tab_label, text="Load images", command=self.label_load_images).grid(column=1,row=1,sticky="w")
        self.label_listbox = tk.Listbox(self.tab_label, width=40, height=20, selectmode=tk.EXTENDED)
        self.label_listbox.grid(column=0,row=2, rowspan=8, sticky="ns")
        self.label_listbox.bind("<<ListboxSelect>>", self.label_on_select)
        self.preview_canvas = tk.Canvas(self.tab_label, width=360, height=360, bg="#222")
        self.preview_canvas.grid(column=1,row=2, rowspan=6)
        navf = ttk.Frame(self.tab_label); navf.grid(column=1,row=8)
        ttk.Button(navf, text="<< Prev", command=self.label_prev).grid(column=0,row=0)
        ttk.Button(navf, text="Next >>", command=self.label_next).grid(column=1,row=0)
        ttk.Button(navf, text="Save annotations", command=self.label_save).grid(column=2,row=0)
        self.attr_frame = ttk.Frame(self.tab_label); self.attr_frame.grid(column=2,row=2)
        ttk.Button(self.tab_label, text="Attribute manager", command=self.open_schema_manager).grid(column=2,row=1)
        ttk.Button(self.tab_label, text="Bulk apply", command=self.label_bulk_apply).grid(column=2,row=3)
        ttk.Button(self.tab_label, text="Export subset", command=self.label_export_subset).grid(column=2,row=4)

        # Train tab
        ttk.Label(self.tab_train, text="Data folder (train/val)").grid(column=0,row=0,sticky="w")
        self.train_data = ttk.Entry(self.tab_train, width=70); self.train_data.insert(0,str(BASE_DIR/"data_final")); self.train_data.grid(column=1,row=0)
        ttk.Label(self.tab_train, text="Models dir").grid(column=0,row=1,sticky="w")
        self.models_dir = ttk.Entry(self.tab_train, width=70); self.models_dir.insert(0,str(BASE_DIR/"models")); self.models_dir.grid(column=1,row=1)
        ttk.Label(self.tab_train, text="Epochs").grid(column=0,row=2,sticky="w")
        self.epochs_e = ttk.Entry(self.tab_train, width=8); self.epochs_e.insert(0,"15"); self.epochs_e.grid(column=1,row=2,sticky="w")
        self.amp_var = tk.BooleanVar(value=True); ttk.Checkbutton(self.tab_train, text="AMP (cuda)", variable=self.amp_var).grid(column=1,row=3,sticky="w")
        ttk.Button(self.tab_train, text="Start training", command=self.run_train_thread).grid(column=1,row=4,sticky="w")

        # Diagnose tab
        ttk.Label(self.tab_diag, text="Model path").grid(column=0,row=0,sticky="w")
        self.diag_model = ttk.Entry(self.tab_diag, width=70); self.diag_model.insert(0,str(BASE_DIR/"models/best_model.pth")); self.diag_model.grid(column=1,row=0)
        ttk.Label(self.tab_diag, text="Validation folder").grid(column=0,row=1,sticky="w")
        self.diag_val = ttk.Entry(self.tab_diag, width=70); self.diag_val.insert(0,str(BASE_DIR/"data_final/val")); self.diag_val.grid(column=1,row=1)
        ttk.Button(self.tab_diag, text="Run diagnose (eval)", command=self.run_diagnose_thread).grid(column=1,row=2,sticky="w")

        # Export tab
        ttk.Label(self.tab_export, text="Model to export").grid(column=0,row=0,sticky="w")
        self.export_model_p = ttk.Entry(self.tab_export, width=70); self.export_model_p.insert(0,str(BASE_DIR/"models/best_model.pth")); self.export_model_p.grid(column=1,row=0)
        ttk.Button(self.tab_export, text="Export TorchScript", command=self.export_torchscript).grid(column=1,row=1,sticky="w")
        ttk.Button(self.tab_export, text="Export ONNX", command=self.export_onnx).grid(column=1,row=2,sticky="w")

        # Serve tab
        ttk.Label(self.tab_serve, text="Model path for serving").grid(column=0,row=0,sticky="w")
        self.serve_model = ttk.Entry(self.tab_serve, width=70); self.serve_model.insert(0,str(BASE_DIR/"models/best_model.pth")); self.serve_model.grid(column=1,row=0)
        ttk.Label(self.tab_serve, text="Labels JSON (optional)").grid(column=0,row=1,sticky="w")
        self.serve_labels = ttk.Entry(self.tab_serve, width=70); self.serve_labels.insert(0,str(BASE_DIR/"labels.json")); self.serve_labels.grid(column=1,row=1)
        ttk.Label(self.tab_serve, text="Port").grid(column=0,row=2,sticky="w")
        self.serve_port = ttk.Entry(self.tab_serve, width=8); self.serve_port.insert(0,"8123"); self.serve_port.grid(column=1,row=2,sticky="w")
        ttk.Button(self.tab_serve, text="Start server", command=self.start_server_thread).grid(column=1,row=3,sticky="w")
        ttk.Button(self.tab_serve, text="Stop server", command=self.stop_server).grid(column=1,row=4,sticky="w")

        # Log area
        self.log_text = tk.Text(frm, width=120, height=14)
        self.log_text.grid(column=0,row=1,columnspan=3, pady=(8,0))
        self.status_var = tk.StringVar(value="idle"); ttk.Label(frm, textvariable=self.status_var).grid(column=0,row=2,sticky="w")
        handler = TextHandler(self.log_text); handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s")); logger.addHandler(handler)

        # internal state
        self.worker = None
        self.gui_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.flask_thread = None

        # labeler internal lists
        self.label_image_folder = None
        self.label_image_paths: List[Path] = []
        self.label_current_index = 0
        self.label_thumbnail_cache: Dict[str, ImageTk.PhotoImage] = {}
        self.label_tk_image_ref = None

        # poll gui queue
        self.root.after(200, self._poll_queue)

    # ------- helpers -------
    def log_message(self, *args):
        line = " ".join(str(a) for a in args) + "\n"
        try:
            self.log_text.configure(state='normal')
            self.log_text.insert('end', line)
            self.log_text.see('end')
            self.log_text.configure(state='disabled')
        except Exception:
            print(line.strip())

    def browse_entry(self, entry):
        p = filedialog.askdirectory()
        if p: entry.delete(0,"end"); entry.insert(0,p)

    # ------- fetch/crop/meta threads -------
    def run_fetch_thread(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Running","Another task is running")
            return
        primary_q = self.fetch_pos.get().strip()
        extras = [l.strip() for l in self.fetch_extra_text.get("1.0", "end").splitlines() if l.strip()]
        queries = [primary_q] + extras
        num = int(self.fetch_num.get().strip())
        out_base = Path(self.fetch_out.get().strip())
        serpapi_key = self.fetch_serp.get().strip() or None
        def job():
            self.status_var.set("fetching")
            safe_mkdir(out_base)
            for q in queries:
                subfolder = out_base / sanitize_foldername(q)
                subfolder.mkdir(parents=True, exist_ok=True)
                logger.info("Fetching query '%s' -> %s", q, subfolder)
                try:
                    if serpapi_key and _HAS_SERPAPI:
                        serpapi_images_for_query(q, subfolder, max_num=num, serpapi_key=serpapi_key)
                    elif _HAS_ICRAWLER:
                        icrawler_download(q, subfolder, max_num=num, engine="google")
                    else:
                        logger.error("No fetch backend available (install serpapi or icrawler)")
                except Exception:
                    logger.exception("Fetch failed for query '%s'", q)
            logger.info("All fetches done -> %s", out_base)
            self.status_var.set("idle")
        self.worker = threading.Thread(target=job, daemon=True); self.worker.start()

    def run_prepare_thread(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Running","Another task is running")
            return
        raw = Path(self.raw_folder.get().strip()); out = Path(self.prep_out.get().strip())
        try:
            rtrain = float(self.r_train.get().strip()); rval = float(self.r_val.get().strip()); rtest = float(self.r_test.get().strip())
        except Exception:
            messagebox.showerror("Invalid ratios", "Train/Val/Test must be floats that sum to 1.0")
            return
        if abs(rtrain + rval + rtest - 1.0) > 1e-6:
            messagebox.showerror("Invalid ratios", "Train+Val+Test must sum to 1.0")
            return
        dedupe_phash = bool(self.phash_var.get())
        phash_thresh = int(self.phash_thresh.get().strip() or 6)
        ext_script = Path(__file__).resolve().parent / "scripts" / "prepare_dataset.py"
        def job():
            self.status_var.set("preparing")
            try:
                if ext_script.exists():
                    cmd = [sys.executable, str(ext_script), "--raw", str(raw), "--out", str(out), "--train", str(rtrain), "--val", str(rval), "--test", str(rtest)]
                    if dedupe_phash:
                        cmd += ["--dedupe_phash", "--phash_threshold", str(phash_thresh)]
                    logger.info("Calling external prepare_dataset script: %s", " ".join(cmd))
                    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    assert p.stdout
                    for line in p.stdout:
                        logger.info("[prepare] %s", line.rstrip())
                    p.wait()
                else:
                    logger.info("Running internal prepare (fallback)")
                    internal_prepare_dataset(raw, out, rtrain, rval, rtest, min_size=(50,50), dedupe_phash=dedupe_phash, phash_threshold=phash_thresh, seed=42)
            except Exception:
                logger.exception("Prepare dataset failed")
            self.status_var.set("idle")
        self.worker = threading.Thread(target=job, daemon=True); self.worker.start()

    def run_crop_thread(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Running","Another task is running")
            return
        raw = Path(self.raw_folder.get().strip()); out = Path(self.faces_out.get().strip())
        margin = float(self.margin_e.get().strip()) if hasattr(self, "margin_e") else 0.2
        minf = int(self.minface_e.get().strip()) if hasattr(self, "minface_e") else 48
        keep = False
        def job():
            self.status_var.set("cropping")
            logger.info("Cropping faces raw=%s out=%s", raw, out)
            stats = crop_faces_in_folder(raw, out, margin=margin, min_size=minf, keep_no_face=keep)
            logger.info("Crop stats: %s", stats)
            self.status_var.set("idle")
        self.worker = threading.Thread(target=job, daemon=True); self.worker.start()

    def run_meta_thread(self):
        raw = Path(self.raw_folder.get().strip()); out = Path(self.faces_out.get().strip())
        meta_path = out / "_crops_meta"
        def job():
            self.status_var.set("metadata")
            try:
                generate_metadata_for_crops(raw, out, meta_path)
                logger.info("Metadata generated at %s(.json/.csv)", meta_path)
            except Exception:
                logger.exception("Metadata generation failed")
            self.status_var.set("idle")
        self.worker = threading.Thread(target=job, daemon=True); self.worker.start()

    # ------- training/diagnose/export/server -------
    def run_train_thread(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Running","Another task is running")
            return
        cfg = {"data": self.train_data.get().strip(), "models_dir": self.models_dir.get().strip(),
               "epochs": int(self.epochs_e.get().strip()), "batch": 32, "lr":1e-4, "freeze_epochs":3,
               "use_sampler": True, "image_size":224, "pretrained": True, "amp": bool(self.amp_var.get()),
               "early_stopping_patience": 5}
        def job():
            self.status_var.set("training")
            stop_ev = threading.Event()
            try:
                run_training(cfg, gui_queue=self.gui_queue, stop_event=stop_ev)
            except Exception:
                logger.exception("Training failed")
            self.status_var.set("idle")
        self.worker = threading.Thread(target=job, daemon=True); self.worker.start()

    def run_diagnose_thread(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Running","Another task is running")
            return
        model_path = self.diag_model.get().strip(); val_dir = Path(self.diag_val.get().strip())
        def job():
            self.status_var.set("diagnosing")
            try:
                labels = sorted([d.name for d in Path(val_dir).parent.iterdir() if d.is_dir()]) if Path(val_dir).parent.exists() else None
                num_classes = len(labels) if labels else 2
                model = build_model(num_classes=num_classes, base_model="resnet50", pretrained=False)
                sd = torch.load(model_path, map_location="cpu")
                if isinstance(sd, dict) and "model_state" in sd:
                    sd = sd["model_state"]
                sd = {k.replace("module.",""):v for k,v in sd.items()}
                model.load_state_dict(sd)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device).eval()
                _, val_tf = make_transforms(224)
                val_ds = datasets.ImageFolder(str(val_dir), transform=val_tf)
                val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=(device.type=="cuda"))
                res = evaluate(model, val_loader, nn.CrossEntropyLoss(), device)
                logger.info("Diagnose result: %s", res)
                messagebox.showinfo("Diagnose", f"val_loss={res['val_loss']:.4f} acc={res['accuracy']:.4f}")
            except Exception:
                logger.exception("Diagnose failed")
            self.status_var.set("idle")
        self.worker = threading.Thread(target=job, daemon=True); self.worker.start()

    def export_torchscript(self):
        model_path = self.export_model_p.get().strip(); out = Path(model_path).parent/ "exported_traced.pt"
        try:
            export_model_torchscript(model_path, str(out), base_model="resnet50", num_classes=2)
            messagebox.showinfo("Export", f"Saved TorchScript to {out}")
        except Exception:
            logger.exception("Export to TorchScript failed")

    def export_onnx(self):
        model_path = self.export_model_p.get().strip(); out = Path(model_path).parent/ "exported.onnx"
        try:
            export_model_onnx(model_path, str(out), base_model="resnet50", num_classes=2)
            messagebox.showinfo("Export", f"Saved ONNX to {out}")
        except Exception:
            logger.exception("Export to ONNX failed")

    def start_server_thread(self):
        if self.flask_thread and self.flask_thread.is_alive():
            messagebox.showinfo("Running","Server already running")
            return
        model_path = self.serve_model.get().strip()
        labels_path = self.serve_labels.get().strip()
        labels = None
        if Path(labels_path).exists():
            try:
                labels = load_json(Path(labels_path))
            except Exception:
                logger.warning("Failed reading labels.json")
        port = int(self.serve_port.get().strip())
        def job():
            try:
                self.status_var.set("serving")
                app = create_flask_app(model_path, labels=labels, base_model="resnet50")
                app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
            except Exception:
                logger.exception("Flask server failed")
            self.status_var.set("idle")
        self.flask_thread = threading.Thread(target=job, daemon=True)
        self.flask_thread.start()
        logger.info("Server starting on port %d", port)

    def stop_server(self):
        if self.flask_thread and self.flask_thread.is_alive():
            messagebox.showinfo("Stop server","Server thread will be left to exit (no safe termination). Restart app to free port if needed.")
        else:
            messagebox.showinfo("Stop server","No running server detected")

    def _poll_queue(self):
        try:
            while True:
                item = self.gui_queue.get_nowait()
                if item["type"]=="epoch":
                    self.status_var.set(f"Epoch {item['epoch']} train_loss={item['train_loss']:.4f} val_loss={item['val_loss']:.4f} acc={item['val_acc']:.3f}")
                elif item["type"]=="done":
                    self.status_var.set(f"Done - best_val={item.get('best_val')}")
        except queue.Empty:
            pass
        self.root.after(200, self._poll_queue)

    # ---------------- Labeler methods ----------------
    def label_load_images(self):
        folder = Path(self.label_folder_e.get().strip())
        if not folder.exists():
            messagebox.showerror("Error", "Folder inválido")
            return
        self.label_image_folder = folder
        exts = {".jpg",".jpeg",".png",".webp",".bmp"}
        paths = [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in exts and p.is_file()]
        self.label_image_paths = paths
        self.label_current_index = 0
        self.label_thumbnail_cache.clear()
        self.label_listbox.delete(0,"end")
        for p in self.label_image_paths:
            self.label_listbox.insert("end", p.name)
        ann_path = folder / "annotations.json"
        self.ann_store.path = ann_path
        if ann_path.exists():
            try:
                self.ann_store.load(ann_path)
                self.log_message(f"Loaded annotations from {ann_path}")
            except Exception:
                logger.exception("Failed loading annotations")
        self.rebuild_label_attr_controls()
        self.label_refresh_preview()

    def label_refresh_preview(self):
        if not self.label_image_paths:
            self.preview_canvas.delete("all")
            return
        p = self.label_image_paths[self.label_current_index]
        key = str(p)
        if key not in self.label_thumbnail_cache:
            pil = make_thumbnail(p, size=(340,340))
            if pil is None:
                pil = Image.new("RGB",(340,340),(50,50,50))
            tkimg = ImageTk.PhotoImage(pil)
            self.label_thumbnail_cache[key] = tkimg
        else:
            tkimg = self.label_thumbnail_cache[key]
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(180,180,image=tkimg)
        self.label_tk_image_ref = tkimg
        ann = self.ann_store.get_annotation(str(p))
        self.label_update_attr_controls(ann)
        try:
            self.label_listbox.selection_clear(0,"end")
            self.label_listbox.selection_set(self.label_current_index)
            self.label_listbox.see(self.label_current_index)
        except Exception:
            pass

    def label_next(self):
        if not self.label_image_paths: return
        self.label_save_current()
        self.label_current_index = min(len(self.label_image_paths)-1, self.label_current_index+1)
        self.label_refresh_preview()

    def label_prev(self):
        if not self.label_image_paths: return
        self.label_save_current()
        self.label_current_index = max(0, self.label_current_index-1)
        self.label_refresh_preview()

    def label_on_select(self, evt=None):
        sel = self.label_listbox.curselection()
        if sel:
            idx = sel[0]
            self.label_current_index = idx
            self.label_refresh_preview()

    def rebuild_label_attr_controls(self):
        for w in self.attr_frame.winfo_children():
            w.destroy()
        self.label_attr_widgets = {}
        r = 0
        for attr, meta in self.schema_mgr.schema.items():
            ttk.Label(self.attr_frame, text=attr).grid(column=0,row=r,sticky="w")
            if meta["type"] == "single":
                var = tk.StringVar(value=meta.get("default",""))
                cb = ttk.Combobox(self.attr_frame, values=meta["labels"], textvariable=var, state="readonly", width=20)
                cb.grid(column=1,row=r,sticky="w")
                self.label_attr_widgets[attr] = {"type":"single","var":var,"widget":cb}
            else:
                box = ttk.Frame(self.attr_frame); box.grid(column=1,row=r,sticky="w")
                vars = {}
                col = 0
                for lab in meta["labels"]:
                    v = tk.BooleanVar(value=False)
                    cb = ttk.Checkbutton(box, text=lab, variable=v)
                    cb.grid(column=col, row=0, sticky="w"); vars[lab]=v; col+=1
                self.label_attr_widgets[attr] = {"type":"multi","vars":vars,"widget":box}
            r += 1
        ttk.Button(self.attr_frame, text="Save current annotation", command=self.label_save_current).grid(column=0,row=r,columnspan=2,pady=(8,0))

    def label_update_attr_controls(self, ann: Dict):
        for attr, meta in self.schema_mgr.schema.items():
            w = self.label_attr_widgets.get(attr)
            if not w: continue
            if meta["type"] == "single":
                val = ann.get(attr, meta.get("default"))
                try: w["var"].set(val if val is not None else "")
                except Exception: pass
            else:
                cur = set(ann.get(attr, []))
                for lab, var in w["vars"].items():
                    var.set(lab in cur)

    def label_gather_current_annotation(self) -> Dict:
        ann = {}
        for attr, meta in self.schema_mgr.schema.items():
            w = self.label_attr_widgets.get(attr)
            if not w: continue
            if meta["type"] == "single":
                val = w["var"].get()
                ann[attr] = val
            else:
                vals = [lab for lab, var in w["vars"].items() if var.get()]
                ann[attr] = vals
        return ann

    def label_save_current(self):
        if not self.label_image_paths: return
        p = self.label_image_paths[self.label_current_index]
        ann = self.label_gather_current_annotation()
        self.ann_store.set_annotation(str(p), ann)
        try:
            self.ann_store.save(self.ann_store.path)
            self.log_message(f"Saved annotation for {p.name}")
        except Exception:
            logger.exception("Failed saving annotation")

    def label_save(self):
        if not self.ann_store.path:
            p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON","*.json")])
            if not p: return
            self.ann_store.path = Path(p)
        self.ann_store.save(self.ann_store.path)
        self.log_message(f"Annotations saved to {self.ann_store.path}")

    def open_schema_manager(self):
        dlg = SchemaDialog(self.root, self.schema_mgr)
        self.root.wait_window(dlg.top)
        if dlg.modified:
            self.rebuild_label_attr_controls()
            self.log_message("Schema updated")

    def label_bulk_apply(self):
        if not self.label_image_paths: return
        attrs = list(self.schema_mgr.schema.keys())
        attr = simpledialog.askstring("Bulk apply", f"Attr a modificar (available: {', '.join(attrs)})")
        if not attr or attr not in self.schema_mgr.schema:
            return
        meta = self.schema_mgr.schema[attr]
        if meta["type"] == "single":
            val = simpledialog.askstring("Value", f"Valor (choices: {', '.join(meta['labels'])})")
            if val is None: return
            images = [str(self.label_image_paths[i]) for i in self.label_listbox.curselection()] or [str(self.label_image_paths[self.label_current_index])]
            self.ann_store.bulk_apply(images, attr, val, self.schema_mgr)
            self.log_message(f"Applied {attr}={val} to {len(images)} images")
        else:
            val = simpledialog.askstring("Value", f"Label to toggle/add (choices: {', '.join(meta['labels'])})")
            if val is None: return
            images = [str(self.label_image_paths[i]) for i in self.label_listbox.curselection()] or [str(self.label_image_paths[self.label_current_index])]
            self.ann_store.bulk_apply(images, attr, val, self.schema_mgr)
            self.log_message(f"Toggled {val} in {attr} for {len(images)} images")
        self.ann_store.save(self.ann_store.path)

    def label_export_subset(self):
        mode = simpledialog.askstring("Export subset", "Export mode: 'selection' or 'all'", initialvalue="selection")
        if not mode: return
        if mode not in ("selection","all"):
            messagebox.showerror("Invalid", "Choose selection or all")
            return
        if mode == "selection":
            sel = self.label_listbox.curselection()
            if not sel: messagebox.showinfo("No selection","Selecciona imágenes"); return
            image_indices = [i for i in sel]
        else:
            image_indices = list(range(len(self.label_image_paths)))
        target = filedialog.askdirectory(title="Select export target folder")
        if not target: return
        target = Path(target)
        org_attr = simpledialog.askstring("Organize by", "Organize by attribute (or leave blank for flat copy)", initialvalue="")
        copied = 0
        for idx in image_indices:
            p = self.label_image_paths[idx]
            ann = self.ann_store.get_annotation(str(p))
            if org_attr and org_attr in ann:
                val = ann[org_attr]
                if isinstance(val, list):
                    for v in val:
                        dest = target / org_attr / str(v); dest.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(p, dest / p.name); copied += 1
                else:
                    dest = target / org_attr / str(val); dest.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, dest / p.name); copied += 1
            else:
                dest = target / "images"; dest.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest / p.name); copied += 1
        messagebox.showinfo("Export done", f"Copied {copied} files to {target}")
        self.log_message(f"Exported {copied} files to {target}")

# ---------------------------
# Text logging handler for Tk Text widget
# ---------------------------
class TextHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget
    def emit(self, record):
        try:
            msg = self.format(record) + "\n"
            self.text_widget.configure(state='normal')
            self.text_widget.insert('end', msg)
            self.text_widget.see('end')
            self.text_widget.configure(state='disabled')
        except Exception:
            pass

# ---------------------------
# CLI entrypoint
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    if args.gui:
        if not _HAS_TK:
            print("Tkinter not available. Install tkinter or run CLI steps.")
            return
        app = PipelineGUI()
        app.root.mainloop()
    else:
        print("Run with --gui to open the interface.")

if __name__ == "__main__":
    main()
