#!/usr/bin/env python3
"""
Entrenamiento con transfer learning y AMP + mejoras (versión corregida).

Mejoras aplicadas tras tus logs:
 - GradScaler creado con (enabled, device) para evitar FutureWarning.
 - Convierte todas las imágenes a RGB en los transforms para evitar errores por paleta/alpha.
 - Habilita ImageFile.LOAD_TRUNCATED_IMAGES para tolerar archivos truncados.
 - Reporte de distribución de clases, opción de class_weights si no se usa sampler.
 - Resto de funcionalidades: AMP, freeze/unfreeze, sampler, checkpoints, resume, early stopping, TB.
"""
from __future__ import annotations

import argparse
import os
import time
import json
import math
import warnings
from pathlib import Path
from typing import Optional, Dict
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # tolerar imágenes truncadas/parciales

# Optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
    _HAS_TB = True
except Exception:
    _HAS_TB = False

# -------- utils --------
def seed_everything(seed: int = 42):
    import random, os
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

def make_transforms(image_size: int = 224):
    # Convertir siempre a RGB al inicio para evitar problemas con paletas / imágenes en BW / alpha
    to_rgb = transforms.Lambda(lambda img: img.convert('RGB') if isinstance(img, Image.Image) else img)
    train_transform = transforms.Compose([
        to_rgb,
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_transform = transforms.Compose([
        to_rgb,
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return train_transform, val_transform

def build_model(num_classes: int, base_model: str = "resnet50", pretrained: bool = True) -> nn.Module:
    if base_model == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    elif base_model == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError("base_model no soportado")
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model

def get_parameter_groups(model: nn.Module, weight_decay: float = 1e-4):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() == 1 or name.endswith(".bias") or "bn" in name.lower() or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]

def create_sampler_if_needed(dataset: datasets.ImageFolder, imbalance_ratio_threshold: float = 1.5) -> Optional[WeightedRandomSampler]:
    targets = np.array([s[1] for s in dataset.samples], dtype=int)
    if len(targets) == 0:
        return None
    counts = np.bincount(targets, minlength=len(dataset.classes))
    counts = np.where(counts == 0, 1, counts)
    ratio = counts.max() / counts.min()
    if ratio <= imbalance_ratio_threshold:
        return None
    class_weights = 1.0 / counts
    sample_weights = class_weights[targets]
    sample_weights = torch.tensor(sample_weights, dtype=torch.double)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler

# -------- training / evaluation loops --------
def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion, optimizer, device,
                    scaler: GradScaler, use_amp: bool, max_grad_norm: Optional[float] = None) -> (float, int):
    model.train()
    running_loss = 0.0
    n_samples = 0
    use_autocast = use_amp and (device.type == 'cuda')
    for inputs, labels in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        ctx = autocast() if use_autocast else nullcontext()
        with ctx:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        if max_grad_norm is not None:
            try:
                scaler.unscale_(optimizer)
            except Exception:
                pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        bs = inputs.size(0)
        running_loss += loss.item() * bs
        n_samples += bs
    avg_loss = running_loss / max(1, n_samples)
    return avg_loss, n_samples

def evaluate(model: nn.Module, dataloader: DataLoader, criterion, device, use_amp: bool) -> Dict:
    model.eval()
    val_loss = 0.0
    y_true = []
    y_pred = []
    n = 0
    use_autocast = use_amp and (device.type == 'cuda')
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            ctx = autocast() if use_autocast else nullcontext()
            with ctx:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            n += inputs.size(0)
    acc = accuracy_score(y_true, y_pred) if len(y_true) else 0.0
    class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0) if len(y_true) else {}
    confmat = confusion_matrix(y_true, y_pred) if len(y_true) else None
    return {
        "val_loss": val_loss / max(1, n),
        "n": n,
        "accuracy": acc,
        "report": class_report,
        "confusion_matrix": confmat,
        "y_true": y_true,
        "y_pred": y_pred
    }

# -------- main --------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/final')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--models_dir', default='models')
    parser.add_argument('--freeze_epochs', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--use_sampler', action='store_true')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=None)
    parser.add_argument('--scheduler', choices=['reduce_on_plateau','cosine'], default='reduce_on_plateau')
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--base_model', type=str, default='resnet50', choices=['resnet50','resnet18'])
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--amp', action='store_true', help='Usar AMP (recomendado si tienes CUDA)')
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)
    os.makedirs(args.models_dir, exist_ok=True)

    train_tf, val_tf = make_transforms(args.image_size)
    train_dir = os.path.join(args.data, 'train')
    val_dir = os.path.join(args.data, 'val')
    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(f"Train/Val folders no encontrados en {args.data}")

    cpu_count = os.cpu_count() or 1
    if os.name == 'nt':
        default_workers = 0
    else:
        default_workers = min(max(1, cpu_count - 1), args.num_workers)
    num_workers = default_workers

    train_dataset = datasets.ImageFolder(train_dir, transform=train_tf)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_tf)

    print("Clases detectadas (train):", train_dataset.classes)
    counts = {}
    for _, t in train_dataset.samples:
        counts.setdefault(t, 0)
        counts[t] += 1
    print("Distribución por índice de clase (train):", counts)

    if len(train_dataset.classes) != args.num_classes:
        print(f"[Warning] num_classes argumento ({args.num_classes}) != clases encontradas en train ({len(train_dataset.classes)}). Usando {len(train_dataset.classes)}")
        args.num_classes = len(train_dataset.classes)

    sampler = None
    if args.use_sampler:
        sampler = create_sampler_if_needed(train_dataset)
        if sampler is not None:
            print("Usando WeightedRandomSampler para corregir desequilibrio de clases")
        else:
            print("Sampler no necesario (clases relativamente balanceadas)")

    pin_memory = True if device.type == 'cuda' else False

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=(sampler is None),
                              sampler=sampler, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    model = build_model(args.num_classes, base_model=args.base_model, pretrained=args.pretrained).to(device)

    # Freeze backbone initially (except fc)
    for name, param in model.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False

    # criterion: if no sampler and desbalance, compute class_weights
    use_class_weights = False
    criterion = None
    if sampler is None:
        # compute counts and maybe use class weights
        target_counts = np.bincount([s[1] for s in train_dataset.samples], minlength=len(train_dataset.classes))
        if target_counts.max() / (target_counts.min() + 1e-12) > 1.5:
            class_weights = torch.tensor((1.0 / (target_counts + 1e-12)) * (target_counts.sum()), dtype=torch.float)
            class_weights = class_weights.to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            use_class_weights = True
            print("Usando class_weights en CrossEntropyLoss debido a desequilibrio de clases.")
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    param_groups = get_parameter_groups(model, weight_decay=args.weight_decay)
    optimizer = AdamW(param_groups, lr=args.lr)

    # scheduler (compatibilidad con versiones antiguas que no soportan verbose)
    if args.scheduler == 'reduce_on_plateau':
        try:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        except TypeError:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    # decide uso de AMP
    use_amp = args.amp and (device.type == 'cuda')
    # crear GradScaler con enabled + device para evitar FutureWarning en algunas versiones
    scaler = GradScaler(enabled=use_amp, device=(device if use_amp else None))

    start_epoch = 1
    best_val = float('inf')
    best_epoch = -1
    history = []

    tb_writer = SummaryWriter(log_dir=os.path.join(args.models_dir, "tb")) if args.tensorboard and _HAS_TB else None
    if args.tensorboard and not _HAS_TB:
        print("TensorBoard no disponible")

    # resume if requested
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        try:
            model_state = ckpt.get('model_state', ckpt)
            model.load_state_dict(model_state)
            optimizer_state = ckpt.get('optimizer_state', None)
            if optimizer_state is not None:
                optimizer.load_state_dict(optimizer_state)
            start_epoch = ckpt.get('epoch', 1) + 1
            best_val = ckpt.get('best_val_loss', best_val)
            print(f"Reanudando desde {args.resume} -> start_epoch={start_epoch}")
        except Exception as e:
            print("Warning al cargar checkpoint:", e)

    epochs_no_improve = 0

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        if epoch == args.freeze_epochs + 1:
            print("-> Unfreezing backbone para fine-tuning completo")
            for param in model.parameters():
                param.requires_grad = True
            param_groups = get_parameter_groups(model, weight_decay=args.weight_decay)
            optimizer = AdamW(param_groups, lr=args.lr * 0.1)
            if args.scheduler == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - epoch + 1))
            # re-create scaler to reflect use_amp/device (some versions recommend reinit)
            scaler = GradScaler(enabled=use_amp, device=(device if use_amp else None))

        train_loss, n_tr = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, use_amp, max_grad_norm=args.grad_clip)
        eval_res = evaluate(model, val_loader, criterion, device, use_amp)
        val_loss = eval_res["val_loss"]
        val_acc = eval_res["accuracy"]

        if args.scheduler == 'reduce_on_plateau':
            try:
                scheduler.step(val_loss)
            except Exception:
                scheduler.step()
        else:
            scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} - train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_acc={val_acc:.4f} time={elapsed:.1f}s")

        if tb_writer:
            tb_writer.add_scalar("loss/train", train_loss, epoch)
            tb_writer.add_scalar("loss/val", val_loss, epoch)
            tb_writer.add_scalar("metrics/val_acc", val_acc, epoch)
            try:
                lr_cur = optimizer.param_groups[0].get("lr", None)
                if lr_cur is not None:
                    tb_writer.add_scalar("lr", lr_cur, epoch)
            except Exception:
                pass

        ckpt_path = os.path.join(args.models_dir, f"ckpt_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_val_loss': best_val
        }, ckpt_path)

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_path = os.path.join(args.models_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model (epoch {epoch}) -> {best_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        row = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "elapsed": float(elapsed)
        }
        history.append(row)
        meta_path = os.path.join(args.models_dir, f"metrics_epoch_{epoch}.json")
        metrics_to_save = {
            "epoch": epoch,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "classification_report": eval_res["report"],
            "confusion_matrix": eval_res["confusion_matrix"].tolist() if eval_res["confusion_matrix"] is not None else None
        }
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metrics_to_save, f, indent=2)
        except Exception as e:
            print("Warning: no se pudo escribir metrics file:", e)

        if args.early_stopping_patience > 0 and epochs_no_improve >= args.early_stopping_patience:
            print(f"No mejora en {epochs_no_improve} epochs (patience={args.early_stopping_patience}). Parando temprano.")
            break

    hist_df = pd.DataFrame(history)
    try:
        hist_df.to_csv(os.path.join(args.models_dir, "training_history.csv"), index=False)
    except Exception as e:
        print("Warning: no se pudo escribir training_history.csv:", e)
    if tb_writer:
        tb_writer.close()

    print("Entrenamiento finalizado. Mejor epoch:", best_epoch, "best_val_loss:", best_val)

if __name__ == "__main__":
    main()
