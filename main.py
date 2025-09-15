#!/usr/bin/env python3
"""
pipeline_all_in_one.py

Unifica: fetch (SerpAPI/icrawler), preprocess (crop faces, normalize, dedup), labeler (atributos discriminativos),
train (robust AMP/resume), diagnose, export (TorchScript/ONNX) y serve (Flask) con una GUI para orquestarlo todo.

Usage:
    python pipeline_all_in_one.py --gui

Notas:
- Comprueba disponibilidad de librerías y hace fallback cuando faltan.
- Guarda anotaciones en <crop_folder>/annotations.json por defecto.
"""
from __future__ import annotations

import argparse
import base64
import contextlib
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
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError, ImageTk
ImageFile.LOAD_TRUNCATED_IMAGES = True

# optional libs detection
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
import requests
from urllib.parse import urlsplit

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("pipeline_all_in_one")

# ---------------------------
# Constants & defaults
# ---------------------------
BASE_DIR = Path(__file__).resolve().parent
BLACKLIST_FILE = BASE_DIR / "blacklist_domains.txt"
FETCH_INDEX = BASE_DIR / ".fetch_index.json"

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
                with open(BLACKLIST_FILE, "a", encoding="utf-8") as f:
                    f.write(domain + "\n")
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
            with open(BLACKLIST_FILE, "a", encoding="utf-8") as f:
                f.write(domain + "\n")
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
                target = dest_dir / "noface"
                safe_mkdir(target)
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
                out_dir = Path(dest_dir)
                safe_mkdir(out_dir)
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
# Dedup (phash) utility (optional)
# ---------------------------
def compute_phash(path: Path):
    if not _HAS_IMGHASH:
        return None
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            ph = imagehash.phash(im)
            return str(ph)
    except Exception:
        return None

# ---------------------------
# Training utilities (compacted)
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
    class_sample_count = np.array([targets.count(t) for t in classes])
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
    scheduler = None
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
                new_state[k.replace("module.","")] = v
            model.load_state_dict(new_state)
            try:
                opt.load_state_dict(ckpt.get("optimizer_state", {}))
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
        if scheduler is not None and cfg.get("scheduler","reduce_on_plateau")=="reduce_on_plateau":
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
# Simple Labeler: schema + annotations
# ---------------------------
DEFAULT_SCHEMA = {
    "pose": {"type": "single", "labels": ["front","profile","three_quarter"], "default":"front"},
    "age": {"type": "single", "labels": ["young","adult","old"], "default":"adult"},
    "hair_color": {"type":"single","labels":["black","brown","blonde","red","gray","other"], "default":"brown"},
    "eye_color": {"type":"single","labels":["brown","blue","green","other"], "default":"brown"},
    "accessories": {"type":"multi","labels":["glasses","hat","earrings","mask"], "default":[]},
    "smiling": {"type":"single","labels":["no","yes"], "default":"no"}
}

class SchemaManager:
    def __init__(self, path: Optional[Path]=None):
        self.path = Path(path) if path else None
        self.schema: Dict[str, Dict] = DEFAULT_SCHEMA.copy()
    def load(self, path: Optional[Path]=None):
        p = Path(path) if path else self.path
        if p and p.exists():
            with open(p, "r", encoding="utf-8") as f:
                self.schema = json.load(f)
        return self.schema
    def save(self, path: Optional[Path]=None):
        p = Path(path) if path else self.path
        if not p:
            raise ValueError("No path provided to save schema")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.schema, f, indent=2, ensure_ascii=False)
    def add_attribute(self, name: str, typ: str="single", labels: Optional[List[str]]=None, default: Any=None):
        if labels is None: labels=[]
        self.schema[name] = {"type": typ, "labels": labels, "default": default if default is not None else ([] if typ=="multi" else (labels[0] if labels else None))}
    def add_label(self, attr: str, label: str):
        if attr not in self.schema: raise KeyError(attr)
        if label not in self.schema[attr]["labels"]: self.schema[attr]["labels"].append(label)

class AnnotationStore:
    def __init__(self, annotations_path: Optional[Path]=None):
        self.path = Path(annotations_path) if annotations_path else None
        self.data: Dict[str, Dict] = {}
    def load(self, path: Optional[Path]=None):
        p = Path(path) if path else self.path
        if p and p.exists():
            with open(p, "r", encoding="utf-8") as f:
                raw = json.load(f)
                if isinstance(raw, dict):
                    self.data = raw
                else:
                    self.data = {r["image"]: r for r in raw}
        return self.data
    def save(self, path: Optional[Path]=None):
        p = Path(path) if path else self.path
        if not p: raise ValueError("No path to save annotations")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)
        try:
            rows = []
            for img, ann in self.data.items():
                row = {"image": img}
                for k, v in ann.items():
                    if isinstance(v, list):
                        row[k] = ";".join([str(x) for x in v])
                    else:
                        row[k] = v
                rows.append(row)
            df = pd.DataFrame(rows)
            df.to_csv(p.with_suffix(".csv"), index=False)
        except Exception:
            pass
    def set_annotation(self, image_path: str, ann: Dict):
        self.data[image_path] = ann
    def get_annotation(self, image_path: str) -> Dict:
        return self.data.get(image_path, {})
    def bulk_apply(self, images: List[str], attr: str, value: Any, schema: SchemaManager):
        typ = schema.schema[attr]["type"]
        for img in images:
            ann = self.get_annotation(img).copy()
            if typ == "single":
                ann[attr] = value
            else:
                cur = set(ann.get(attr, []))
                if isinstance(value, list):
                    cur.update(value)
                else:
                    if value in cur: cur.remove(value)
                    else: cur.add(value)
                ann[attr] = sorted(list(cur))
            self.set_annotation(img, ann)

# thumbnail helper
def make_thumbnail(path: Path, size=(320,320)):
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail(size, Image.LANCZOS)
            return im.copy()
    except Exception:
        return None

# ---------------------------
# Flask serve (same as before)
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
                if not b64: return jsonify({"error":"no image"}), 400
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
# GUI orchestrator (integrates labeler tab)
# ---------------------------
class PipelineGUI:
    def __init__(self):
        if not _HAS_TK:
            raise RuntimeError("Tkinter not available")
        self.root = tk.Tk()
        self.root.title("Unified Pipeline GUI")
        frm = ttk.Frame(self.root, padding=8); frm.grid(sticky="nsew")
        nb = ttk.Notebook(frm); nb.grid(column=0,row=0, sticky="nsew")
        # tabs
        self.tab_fetch = ttk.Frame(nb); self.tab_prep = ttk.Frame(nb); self.tab_label = ttk.Frame(nb)
        self.tab_train = ttk.Frame(nb); self.tab_diag = ttk.Frame(nb); self.tab_export = ttk.Frame(nb); self.tab_serve = ttk.Frame(nb)
        nb.add(self.tab_fetch, text="Fetch"); nb.add(self.tab_prep, text="Prep"); nb.add(self.tab_label, text="Labeler")
        nb.add(self.tab_train, text="Train"); nb.add(self.tab_diag, text="Diagnose"); nb.add(self.tab_export, text="Export"); nb.add(self.tab_serve, text="Serve")

        # Fetch tab
        ttk.Label(self.tab_fetch, text="Positive query").grid(column=0,row=0,sticky="w")
        self.fetch_pos = ttk.Entry(self.tab_fetch, width=50); self.fetch_pos.insert(0,"Linda Hamilton"); self.fetch_pos.grid(column=1,row=0)
        ttk.Label(self.tab_fetch, text="Num images").grid(column=0,row=1,sticky="w")
        self.fetch_num = ttk.Entry(self.tab_fetch, width=10); self.fetch_num.insert(0,"500"); self.fetch_num.grid(column=1,row=1,sticky="w")
        ttk.Label(self.tab_fetch, text="Out folder").grid(column=0,row=2,sticky="w")
        self.fetch_out = ttk.Entry(self.tab_fetch, width=60); self.fetch_out.insert(0,str(BASE_DIR/"data_fetch")); self.fetch_out.grid(column=1,row=2)
        ttk.Label(self.tab_fetch, text="SerpAPI key (optional)").grid(column=0,row=3,sticky="w")
        self.fetch_serp = ttk.Entry(self.tab_fetch, width=60); self.fetch_serp.grid(column=1,row=3)
        ttk.Button(self.tab_fetch, text="Run fetch", command=self.run_fetch_thread).grid(column=1,row=4,sticky="w")

        # Prep tab
        ttk.Label(self.tab_prep, text="Raw folder").grid(column=0,row=0,sticky="w")
        self.raw_folder = ttk.Entry(self.tab_prep, width=70); self.raw_folder.insert(0,str(BASE_DIR/"data_fetch")); self.raw_folder.grid(column=1,row=0)
        ttk.Button(self.tab_prep, text="Browse", command=lambda: self.browse_entry(self.raw_folder)).grid(column=2,row=0)
        ttk.Label(self.tab_prep, text="Faces out").grid(column=0,row=1,sticky="w")
        self.faces_out = ttk.Entry(self.tab_prep, width=70); self.faces_out.insert(0,str(BASE_DIR/"data_faces")); self.faces_out.grid(column=1,row=1)
        ttk.Button(self.tab_prep, text="Browse", command=lambda: self.browse_entry(self.faces_out)).grid(column=2,row=1)
        ttk.Label(self.tab_prep, text="Margin").grid(column=0,row=2,sticky="w")
        self.margin_e = ttk.Entry(self.tab_prep, width=8); self.margin_e.insert(0,"0.2"); self.margin_e.grid(column=1,row=2,sticky="w")
        ttk.Label(self.tab_prep, text="Min face size").grid(column=0,row=3,sticky="w")
        self.minface_e = ttk.Entry(self.tab_prep, width=8); self.minface_e.insert(0,"48"); self.minface_e.grid(column=1,row=3,sticky="w")
        self.keep_no_face = tk.BooleanVar(value=False); ttk.Checkbutton(self.tab_prep, text="Keep no-face copies", variable=self.keep_no_face).grid(column=1,row=4,sticky="w")
        ttk.Button(self.tab_prep, text="Crop faces", command=self.run_crop_thread).grid(column=1,row=5,sticky="w")
        ttk.Button(self.tab_prep, text="Generate metadata CSV/JSON", command=self.run_meta_thread).grid(column=1,row=6,sticky="w")

        # Labeler tab (integrated)
        self.schema_mgr = SchemaManager()
        self.ann_store = AnnotationStore()
        ttk.Label(self.tab_label, text="Faces folder").grid(column=0,row=0,sticky="w")
        self.label_folder_e = ttk.Entry(self.tab_label, width=60); self.label_folder_e.insert(0,str(BASE_DIR/"data_faces")); self.label_folder_e.grid(column=1,row=0)
        ttk.Button(self.tab_label, text="Browse", command=lambda: self.browse_entry(self.label_folder_e)).grid(column=2,row=0)
        ttk.Button(self.tab_label, text="Load images", command=self.label_load_images).grid(column=1,row=1,sticky="w")
        # left: list
        self.label_listbox = tk.Listbox(self.tab_label, width=40, height=20, selectmode=tk.EXTENDED)
        self.label_listbox.grid(column=0,row=2, rowspan=6, sticky="ns")
        self.label_listbox.bind("<<ListboxSelect>>", self.label_on_select)
        # center preview
        self.preview_canvas = tk.Canvas(self.tab_label, width=360, height=360, bg="#222")
        self.preview_canvas.grid(column=1,row=2, rowspan=6)
        navf = ttk.Frame(self.tab_label); navf.grid(column=1,row=8)
        ttk.Button(navf, text="<< Prev", command=self.label_prev).grid(column=0,row=0)
        ttk.Button(navf, text="Next >>", command=self.label_next).grid(column=1,row=0)
        ttk.Button(navf, text="Save annotations", command=self.label_save).grid(column=2,row=0)
        # right: attributes
        self.attr_frame = ttk.Frame(self.tab_label); self.attr_frame.grid(column=2,row=2)
        ttk.Button(self.tab_label, text="Attribute manager", command=self.open_schema_manager).grid(column=2,row=1)
        ttk.Button(self.tab_label, text="Bulk apply", command=self.label_bulk_apply).grid(column=2,row=3)
        ttk.Button(self.tab_label, text="Export subset", command=self.label_export_subset).grid(column=2,row=4)
        # bottom: log
        self.log = tk.Text(frm, width=120, height=14); self.log.grid(column=0,row=2,columnspan=3, pady=(8,0))
        self.status_var = tk.StringVar(value="idle"); ttk.Label(frm, textvariable=self.status_var).grid(column=0,row=3,sticky="w")
        handler = TextHandler(self.log); handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s")); logger.addHandler(handler)

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
    def log(self, *args):
        line = " ".join(str(a) for a in args) + "\n"
        try:
            self.log.insert("end", line)
            self.log.see("end")
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
        q = self.fetch_pos.get().strip(); num = int(self.fetch_num.get().strip()); out = Path(self.fetch_out.get().strip())
        serpapi_key = self.fetch_serp.get().strip() or None
        def job():
            self.status_var.set("fetching")
            safe_mkdir(out)
            try:
                if serpapi_key and _HAS_SERPAPI:
                    serpapi_images_for_query(q, out, max_num=num, serpapi_key=serpapi_key)
                elif _HAS_ICRAWLER:
                    icrawler_download(q, out, max_num=num, engine="google")
                else:
                    logger.error("No fetch backend available (install serpapi or icrawler)")
                logger.info("Fetch done -> %s", out)
            except Exception:
                logger.exception("Fetch failed")
            self.status_var.set("idle")
        self.worker = threading.Thread(target=job, daemon=True); self.worker.start()

    def run_crop_thread(self):
        if self.worker and self.worker.is_alive():
            messagebox.showinfo("Running","Another task is running")
            return
        raw = Path(self.raw_folder.get().strip()); out = Path(self.faces_out.get().strip())
        margin = float(self.margin_e.get().strip()); minf = int(self.minface_e.get().strip()); keep = bool(self.keep_no_face.get())
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
        # set annotations path default
        ann_path = folder / "annotations.json"
        self.ann_store.path = ann_path
        if ann_path.exists():
            try:
                self.ann_store.load(ann_path)
                self.log(f"Loaded annotations from {ann_path}")
            except Exception:
                logger.exception("Failed loading annotations")
        # rebuild attribute controls based on schema
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
        # update attr widgets from annotation
        ann = self.ann_store.get_annotation(str(p))
        self.label_update_attr_controls(ann)
        # select in listbox
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
            self.log(f"Saved annotation for {p.name}")
        except Exception:
            logger.exception("Failed saving annotation")

    def label_save(self):
        if not self.ann_store.path:
            p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON","*.json")])
            if not p: return
            self.ann_store.path = Path(p)
        self.ann_store.save(self.ann_store.path)
        self.log(f"Annotations saved to {self.ann_store.path}")

    def open_schema_manager(self):
        dlg = SchemaDialog(self.root, self.schema_mgr)
        self.root.wait_window(dlg.top)
        if dlg.modified:
            self.rebuild_label_attr_controls()
            self.log("Schema updated")

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
            self.log(f"Applied {attr}={val} to {len(images)} images")
        else:
            val = simpledialog.askstring("Value", f"Label to toggle/add (choices: {', '.join(meta['labels'])})")
            if val is None: return
            images = [str(self.label_image_paths[i]) for i in self.label_listbox.curselection()] or [str(self.label_image_paths[self.label_current_index])]
            self.ann_store.bulk_apply(images, attr, val, self.schema_mgr)
            self.log(f"Toggled {val} in {attr} for {len(images)} images")
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
        self.log(f"Exported {copied} files to {target}")

# Schema dialog used by labeler
class SchemaDialog:
    def __init__(self, parent, schema_mgr: SchemaManager):
        self.parent = parent; self.schema_mgr = schema_mgr; self.modified = False
        self.top = tk.Toplevel(parent); self.top.title("Attribute manager")
        frm = ttk.Frame(self.top, padding=8); frm.grid()
        self.tree = ttk.Treeview(frm, columns=("type","labels"), show="headings", height=10)
        self.tree.heading("type", text="Type"); self.tree.heading("labels", text="Labels"); self.tree.grid(column=0,row=0,columnspan=3)
        self.refresh_tree()
        ttk.Button(frm, text="Add attribute", command=self.add_attribute).grid(column=0,row=1)
        ttk.Button(frm, text="Add label", command=self.add_label).grid(column=1,row=1)
        ttk.Button(frm, text="Save schema", command=self.save_schema).grid(column=2,row=1)
        ttk.Button(frm, text="Close", command=self.close).grid(column=1,row=2)
    def refresh_tree(self):
        for i in self.tree.get_children(): self.tree.delete(i)
        for k,v in self.schema_mgr.schema.items():
            typ = v.get("type","single"); labs = ",".join(v.get("labels",[]))
            self.tree.insert("", "end", iid=k, values=(typ, labs))
    def add_attribute(self):
        name = simpledialog.askstring("Attribute name", "New attribute name", parent=self.top)
        if not name: return
        typ = simpledialog.askstring("Type", "Type: 'single' or 'multi'", parent=self.top, initialvalue="single")
        if typ not in ("single","multi"): messagebox.showerror("Invalid","Type must be single or multi"); return
        labels = simpledialog.askstring("Labels", "Comma-separated labels (example: a,b,c)", parent=self.top)
        labs = [s.strip() for s in labels.split(",")] if labels else []
        default = labs[0] if typ=="single" and labs else ([] if typ=="multi" else None)
        self.schema_mgr.add_attribute(name, typ, labs, default); self.modified=True; self.refresh_tree()
    def add_label(self):
        sel = self.tree.selection()
        if not sel: messagebox.showinfo("Select","Select attribute row first"); return
        attr = sel[0]
        lab = simpledialog.askstring("New label", f"New label to add to {attr}", parent=self.top)
        if not lab: return
        self.schema_mgr.add_label(attr, lab); self.modified=True; self.refresh_tree()
    def save_schema(self):
        p = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON","*.json")])
        if not p: return
        self.schema_mgr.save(Path(p)); messagebox.showinfo("Saved", f"Schema saved to {p}")
    def close(self):
        self.top.destroy()

# Text logging handler
class TextHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__(); self.text_widget = text_widget
    def emit(self, record):
        try:
            msg = self.format(record) + "\n"
            self.text_widget.configure(state="normal")
            self.text_widget.insert("end", msg)
            self.text_widget.see("end")
            self.text_widget.configure(state="disabled")
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
        gui = PipelineGUI()
        gui.root.mainloop()
    else:
        print("Run with --gui to open the interface.")

if __name__ == "__main__":
    main()
