#!/usr/bin/env python3
"""
train_robust_gui.py

Script de entrenamiento robusto con:
 - manejo compatible de AMP (torch.amp) y fallback seguro
 - autocast/GradScaler con compatibilidad hacia atrás
 - resume flexible (carga parcial de checkpoints) y guardado seguro
 - freeze/unfreeze progresivo para fine-tuning
 - WeightedRandomSampler opcional y class weights
 - early stopping, grad clipping, scheduler flexible
 - logging a fichero + consola + GUI
 - Interfaz gráfica (Tkinter) para configurar parámetros, lanzar/monitorizar y parar el run
 - guardado de config JSON y botones para cargar/guardar presets

Uso CLI (sin GUI):
  python scripts/train_robust_gui.py --data data/final --epochs 15 --batch 32 --lr 1e-4 --amp

Uso GUI:
  python scripts/train_robust_gui.py --gui

Nota: El GUI lanza el entrenamiento en un hilo de fondo. Usa el botón STOP para pedir que termine tras la batch actual.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import queue
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# optional TensorBoard
try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
    _HAS_TB = True
except Exception:
    _HAS_TB = False

# GUI
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    _HAS_TK = True
except Exception:
    _HAS_TK = False

# safe import for amp
try:
    from torch.amp import GradScaler, autocast
    _HAS_AMP = True
except Exception:
    _HAS_AMP = False

# ---------------- logging ----------------
logger = logging.getLogger('train_robust')
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(fmt)
logger.addHandler(ch)

# ---------------- helpers ----------------

def seed_everything(seed: int = 42):
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
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform


def build_model(num_classes: int, base_model: str = 'resnet50', pretrained: bool = True) -> nn.Module:
    if base_model == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    elif base_model == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError('base_model no soportado')
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


def get_parameter_groups(model: nn.Module, weight_decay: float = 1e-4):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name.endswith('.bias'):
            no_decay.append(param)
        else:
            decay.append(param)
    return [{'params': decay, 'weight_decay': weight_decay}, {'params': no_decay, 'weight_decay': 0.0}]


def create_sampler_if_needed(dataset: datasets.ImageFolder):
    targets = [s[1] for s in dataset.samples]
    classes = sorted(set(targets))
    class_sample_count = np.array([targets.count(t) for t in classes])
    if class_sample_count.max() / (class_sample_count.min() + 1e-12) > 1.5:
        weights = 1.0 / torch.tensor([class_sample_count[t] for t in targets], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        return sampler
    return None


# Dummy GradScaler for fallback
class DummyGradScaler:
    def scale(self, loss):
        return loss
    def step(self, optimizer):
        optimizer.step()
    def update(self):
        return None
    def unscale_(self, optimizer):
        return None


def create_grad_scaler(use_amp: bool):
    """Create a GradScaler with best-effort compatibility.
    If AMP not available or use_amp=False, return DummyGradScaler.
    """
    if not use_amp or not _HAS_AMP:
        return DummyGradScaler()
    try:
        return GradScaler()
    except Exception:
        return DummyGradScaler()


def autocast_ctx(enabled: bool, device_type: str = 'cuda'):
    """Return an autocast-like context manager compatible with different torch versions."""
    if not enabled or not _HAS_AMP:
        return contextlib.nullcontext()
    try:
        return autocast(device_type)
    except TypeError:
        # some versions accept no args
        try:
            return autocast()
        except Exception:
            return contextlib.nullcontext()

# ---------------- training core ----------------

def train_one_epoch(model: nn.Module, dataloader: DataLoader, criterion, optimizer, device,
                    scaler, stop_event: threading.Event, max_grad_norm: Optional[float] = None, logger: logging.Logger = logger) -> Tuple[float, int]:
    model.train()
    running_loss = 0.0
    n_samples = 0
    for inputs, labels in dataloader:
        if stop_event.is_set():
            logger.info('Stop requested - breaking training epoch')
            break
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with autocast_ctx(isinstance(scaler, GradScaler) and scaler is not None, device.type if hasattr(device, 'type') else 'cpu'):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaled = scaler.scale(loss)
        scaled.backward()
        if max_grad_norm is not None:
            if hasattr(scaler, 'unscale_'):
                try:
                    scaler.unscale_(optimizer)
                except Exception:
                    pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        try:
            scaler.step(optimizer)
            scaler.update()
        except Exception as e:
            # fallback: if scaler fails, try plain step
            logger.warning('Scaler step failed, falling back to plain optimizer.step(): %s', e)
            optimizer.step()
        bs = inputs.size(0)
        running_loss += loss.item() * bs
        n_samples += bs
    return (running_loss / max(1, n_samples), n_samples)


def evaluate(model: nn.Module, dataloader: DataLoader, criterion, device, logger: logging.Logger = logger) -> Dict:
    model.eval()
    val_loss = 0.0
    y_true = []
    y_pred = []
    n = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            n += inputs.size(0)
    acc = accuracy_score(y_true, y_pred) if len(y_true) else 0.0
    class_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    confmat = confusion_matrix(y_true, y_pred) if len(y_true) else None
    return {"val_loss": val_loss / max(1, n), "n": n, "accuracy": acc, "report": class_report, "confusion_matrix": confmat, "y_true": y_true, "y_pred": y_pred}

# ----------------- training orchestrator ----------------

def safe_load_checkpoint(path: str, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None, scaler = None):
    """Attempt flexible loading of checkpoints. Returns (start_epoch, best_val)"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(path, map_location=device)
    start_epoch = 1
    best_val = float('inf')
    # flexible model state
    if 'model_state' in ckpt:
        state = ckpt['model_state']
    elif 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        # assume ckpt is raw state_dict
        state = ckpt
    try:
        model.load_state_dict(state)
        logger.info('Loaded model state from %s', path)
    except Exception as e:
        # try to adapt keys (remove module. etc.)
        new_state = {}
        for k, v in state.items():
            nk = k.replace('module.', '')
            new_state[nk] = v
        try:
            model.load_state_dict(new_state)
            logger.warning('Loaded model state with key mapping from %s', path)
        except Exception as e2:
            logger.exception('Failed to load model state cleanly: %s; %s', e, e2)
    # optimizer
    if optimizer is not None and 'optimizer_state' in ckpt:
        try:
            optimizer.load_state_dict(ckpt['optimizer_state'])
            start_epoch = ckpt.get('epoch', start_epoch) + 1
            best_val = ckpt.get('best_val_loss', best_val)
            logger.info('Loaded optimizer state, resuming from epoch %d', start_epoch)
        except Exception:
            logger.warning('Could not load optimizer state from checkpoint (incompatible)')
    # scaler
    if scaler is not None and 'scaler_state' in ckpt:
        try:
            if hasattr(scaler, 'load_state_dict'):
                scaler.load_state_dict(ckpt['scaler_state'])
        except Exception:
            logger.warning('Could not load scaler state (incompatible)')
    return start_epoch, best_val


def save_checkpoint(path: str, epoch: int, model: nn.Module, optimizer: Optional[torch.optim.Optimizer], best_val_loss: float, scaler=None):
    payload = {'epoch': epoch, 'model_state': model.state_dict(), 'best_val_loss': best_val_loss}
    if optimizer is not None:
        try:
            payload['optimizer_state'] = optimizer.state_dict()
        except Exception:
            logger.warning('Could not serialize optimizer state')
    if scaler is not None and hasattr(scaler, 'state_dict'):
        try:
            payload['scaler_state'] = scaler.state_dict()
        except Exception:
            logger.debug('scaler state not serializable')
    try:
        torch.save(payload, path)
    except Exception as e:
        logger.exception('Failed to save checkpoint %s: %s', path, e)


def run_training(cfg: Dict, gui_queue: Optional[queue.Queue] = None, stop_event: Optional[threading.Event] = None):
    """Runs the training according to cfg dictionary. If gui_queue provided, posts progress messages there."""
    seed_everything(cfg.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.get('use_cuda', True) else 'cpu')
    logger.info('Device: %s', device)

    models_dir = Path(cfg.get('models_dir', 'models'))
    models_dir.mkdir(parents=True, exist_ok=True)

    train_tf, val_tf = make_transforms(cfg.get('image_size', 224))
    train_dir = Path(cfg['data']) / 'train'
    val_dir = Path(cfg['data']) / 'val'
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError('Train/Val folders no encontrados; espera data/train y data/val dentro de --data')

    # tolerant PIL
    try:
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
    except Exception:
        pass

    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_dataset = datasets.ImageFolder(str(val_dir), transform=val_tf)

    logger.info('Clases detectadas (train): %s', train_dataset.classes)

    # sampler
    sampler = None
    if cfg.get('use_sampler', False):
        sampler = create_sampler_if_needed(train_dataset)
        if sampler is not None:
            logger.info('Using WeightedRandomSampler')

    targets = [s[1] for s in train_dataset.samples]
    class_counts = np.array([targets.count(i) for i in range(len(train_dataset.classes))]) if len(targets) else np.array([])
    class_weights = None
    if sampler is None and class_counts.size and (class_counts.max() / (class_counts.min() + 1e-12) > 1.5):
        class_weights = torch.tensor((1.0 / (class_counts + 1e-12)) * (class_counts.sum() / class_counts.sum()), dtype=torch.float)
        logger.info('Applying class weights to loss due to imbalance')

    pin_memory = True if device.type == 'cuda' else False
    train_loader = DataLoader(train_dataset, batch_size=cfg.get('batch', 32), shuffle=(sampler is None), sampler=sampler, num_workers=cfg.get('num_workers', 4), pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=cfg.get('batch', 32), shuffle=False, num_workers=cfg.get('num_workers', 4), pin_memory=pin_memory)

    model = build_model(cfg.get('num_classes', len(train_dataset.classes) if len(train_dataset.classes) else 2), base_model=cfg.get('base_model','resnet50'), pretrained=cfg.get('pretrained', True))
    model = model.to(device)

    # freeze backbone initially (except fc)
    freeze_epochs = cfg.get('freeze_epochs', 3)
    if freeze_epochs > 0:
        for name, param in model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False
        logger.info('Backbone frozen for first %d epochs', freeze_epochs)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
    param_groups = get_parameter_groups(model, weight_decay=cfg.get('weight_decay', 1e-4))
    optimizer = AdamW(param_groups, lr=cfg.get('lr', 1e-4))

    # scheduler
    scheduler_name = cfg.get('scheduler', 'reduce_on_plateau')
    if scheduler_name == 'reduce_on_plateau':
        try:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        except TypeError:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.get('epochs', 20)))

    use_amp = cfg.get('amp', False) and device.type == 'cuda'
    scaler = create_grad_scaler(use_amp)
    if isinstance(scaler, DummyGradScaler):
        logger.info('AMP disabled or not available -> using DummyGradScaler')
    else:
        logger.info('AMP enabled -> using GradScaler')

    start_epoch = 1
    best_val = float('inf')
    history = []

    tb_writer = SummaryWriter(log_dir=str(Path(models_dir) / 'tb')) if cfg.get('tensorboard', False) and _HAS_TB else None

    # resume
    if cfg.get('resume'):
        try:
            s_epoch, bval = safe_load_checkpoint(cfg['resume'], model, optimizer, scaler)
            start_epoch = s_epoch
            best_val = bval
        except Exception as e:
            logger.warning('Resume failed: %s', e)

    epochs_no_improve = 0
    stop_event = stop_event or threading.Event()

    try:
        for epoch in range(start_epoch, cfg.get('epochs', 20) + 1):
            t0 = time.time()
            if epoch == freeze_epochs + 1:
                logger.info('-> Unfreezing backbone for full fine-tuning')
                for param in model.parameters():
                    param.requires_grad = True
                param_groups = get_parameter_groups(model, weight_decay=cfg.get('weight_decay', 1e-4))
                optimizer = AdamW(param_groups, lr=cfg.get('lr', 1e-4) * cfg.get('ft_lr_scale', 0.1))
                if cfg.get('scheduler') == 'cosine':
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.get('epochs', 20) - epoch + 1))
                scaler = create_grad_scaler(use_amp)

            train_loss, n_tr = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, stop_event, max_grad_norm=cfg.get('grad_clip', None), logger=logger)
            eval_res = evaluate(model, val_loader, criterion, device, logger=logger)
            val_loss = eval_res['val_loss']
            val_acc = eval_res['accuracy']

            # scheduler step
            if scheduler_name == 'reduce_on_plateau':
                try:
                    scheduler.step(val_loss)
                except Exception:
                    try:
                        scheduler.step(val_loss)
                    except Exception:
                        pass
            else:
                try:
                    scheduler.step()
                except Exception:
                    pass

            elapsed = time.time() - t0
            logger.info('Epoch %d/%d - train_loss=%.6f val_loss=%.6f val_acc=%.4f time=%.1fs', epoch, cfg.get('epochs', 20), train_loss, val_loss, val_acc, elapsed)

            if tb_writer:
                tb_writer.add_scalar('loss/train', train_loss, epoch)
                tb_writer.add_scalar('loss/val', val_loss, epoch)
                tb_writer.add_scalar('metrics/val_acc', val_acc, epoch)
                try:
                    tb_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
                except Exception:
                    pass

            # save checkpoint
            ckpt_path = str(Path(models_dir) / f'ckpt_epoch_{epoch}.pth')
            save_checkpoint(ckpt_path, epoch, model, optimizer, best_val, scaler=scaler)

            if val_loss < best_val:
                best_val = val_loss
                best_path = str(Path(models_dir) / 'best_model.pth')
                try:
                    torch.save(model.state_dict(), best_path)
                    logger.info('Saved best model (epoch %d) -> %s', epoch, best_path)
                except Exception as e:
                    logger.warning('Could not save best model: %s', e)
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            history.append({'epoch': epoch, 'train_loss': float(train_loss), 'val_loss': float(val_loss), 'val_acc': float(val_acc), 'elapsed': elapsed})

            # save metrics
            meta_path = Path(models_dir) / f'metrics_epoch_{epoch}.json'
            try:
                metrics_to_save = {'epoch': epoch, 'val_loss': val_loss, 'val_acc': val_acc, 'classification_report': eval_res['report'], 'confusion_matrix': eval_res['confusion_matrix'].tolist() if eval_res['confusion_matrix'] is not None else None}
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(metrics_to_save, f, indent=2)
            except Exception:
                logger.warning('Could not save metrics json for epoch %d', epoch)

            # GUI messages
            if gui_queue is not None:
                gui_queue.put({'type': 'epoch', 'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'val_acc': val_acc})

            # early stopping
            if cfg.get('early_stopping_patience', 5) > 0 and epochs_no_improve >= cfg.get('early_stopping_patience', 5):
                logger.info('No improvement in %d epochs. Early stopping.', epochs_no_improve)
                break

            if stop_event.is_set():
                logger.info('Stop requested by user; ending training loop')
                break

    except Exception as e:
        logger.exception('Training failed with exception: %s', e)
    finally:
        # save history
        try:
            pd.DataFrame(history).to_csv(Path(models_dir) / 'training_history.csv', index=False)
        except Exception:
            logger.warning('Could not save training history csv')
        if tb_writer:
            tb_writer.close()
        logger.info('Training finished. Best val loss: %s', best_val)
        if gui_queue is not None:
            gui_queue.put({'type': 'done', 'best_val': best_val})


# ---------------- GUI ----------------
class TrainerGUI:
    def __init__(self):
        if not _HAS_TK:
            raise RuntimeError('Tkinter not available on this system')
        self.root = tk.Tk()
        self.root.title('Train Robust GUI')
        frm = ttk.Frame(self.root, padding=8)
        frm.grid()

        # form entries
        self.entries = {}
        def add_row(label_text, default, row, width=40):
            ttk.Label(frm, text=label_text).grid(column=0, row=row, sticky='w')
            e = ttk.Entry(frm, width=width)
            e.insert(0, str(default))
            e.grid(column=1, row=row, sticky='w')
            self.entries[label_text] = e

        add_row('data', 'data/final', 0)
        add_row('models_dir', 'models', 1)
        add_row('epochs', '20', 2, width=10)
        add_row('batch', '32', 3, width=10)
        add_row('lr', '1e-4', 4, width=10)
        add_row('freeze_epochs', '3', 5, width=10)
        add_row('base_model', 'resnet50', 6, width=15)
        add_row('image_size', '224', 7, width=10)

        self.amp_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text='AMP (cuda)', variable=self.amp_var).grid(column=1, row=8, sticky='w')
        self.pretrained_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frm, text='Pretrained weights', variable=self.pretrained_var).grid(column=1, row=9, sticky='w')

        btn_frame = ttk.Frame(frm)
        btn_frame.grid(column=0, row=10, columnspan=2, pady=(6,6))
        ttk.Button(btn_frame, text='Start', command=self.start).grid(column=0, row=0, padx=4)
        ttk.Button(btn_frame, text='Stop', command=self.stop).grid(column=1, row=0, padx=4)
        ttk.Button(btn_frame, text='Save config', command=self.save_config).grid(column=2, row=0, padx=4)
        ttk.Button(btn_frame, text='Load config', command=self.load_config).grid(column=3, row=0, padx=4)

        self.log = tk.Text(frm, width=100, height=20)
        self.log.grid(column=0, row=11, columnspan=2)

        self.status_var = tk.StringVar(value='idle')
        ttk.Label(frm, textvariable=self.status_var).grid(column=0, row=12, columnspan=2, sticky='w')

        # logging handler to text
        self._orig_handlers = logger.handlers[:]  # keep
        handler = TextHandler(self.log)
        handler.setFormatter(fmt)
        logger.addHandler(handler)

        self.worker_thread = None
        self.gui_queue = queue.Queue()
        self.stop_event = threading.Event()
        # poll gui queue
        self.root.after(200, self._poll_queue)

    def _poll_queue(self):
        try:
            while True:
                item = self.gui_queue.get_nowait()
                if item['type'] == 'epoch':
                    self.status_var.set(f"Epoch {item['epoch']} - train_loss={item['train_loss']:.4f} val_loss={item['val_loss']:.4f} val_acc={item['val_acc']:.3f}")
                elif item['type'] == 'done':
                    self.status_var.set(f"Done - best_val={item.get('best_val')}")
        except queue.Empty:
            pass
        self.root.after(200, self._poll_queue)

    def start(self):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo('Running', 'Already running')
            return
        cfg = self._gather_cfg()
        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=run_training, args=(cfg, self.gui_queue, self.stop_event), daemon=True)
        self.worker_thread.start()
        self.status_var.set('running')

    def stop(self):
        if self.worker_thread and self.worker_thread.is_alive():
            self.stop_event.set()
            self.status_var.set('stop requested')
        else:
            messagebox.showinfo('Not running', 'No active training')

    def _gather_cfg(self):
        cfg = {
            'data': self.entries['data'].get().strip(),
            'models_dir': self.entries['models_dir'].get().strip(),
            'epochs': int(self.entries['epochs'].get().strip()),
            'batch': int(self.entries['batch'].get().strip()),
            'lr': float(self.entries['lr'].get().strip()),
            'freeze_epochs': int(self.entries['freeze_epochs'].get().strip()),
            'base_model': self.entries['base_model'].get().strip(),
            'image_size': int(self.entries['image_size'].get().strip()),
            'amp': bool(self.amp_var.get()),
            'pretrained': bool(self.pretrained_var.get()),
            'use_cuda': True,
            'num_workers': min(max(1, (os.cpu_count() or 1) - 1), 8),
            'use_sampler': True,
            'weight_decay': 1e-4,
            'grad_clip': 1.0,
            'scheduler': 'reduce_on_plateau',
            'early_stopping_patience': 5,
            'tensorboard': False
        }
        return cfg

    def save_config(self):
        cfg = self._gather_cfg()
        p = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('JSON','*.json')])
        if not p:
            return
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2)
        messagebox.showinfo('Saved', f'Config saved to {p}')

    def load_config(self):
        p = filedialog.askopenfilename(filetypes=[('JSON','*.json')])
        if not p:
            return
        with open(p, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        # fill fields we recognize
        for k, e in self.entries.items():
            if k in cfg:
                e.delete(0, 'end')
                e.insert(0, str(cfg[k]))
        if 'amp' in cfg:
            self.amp_var.set(bool(cfg['amp']))
        if 'pretrained' in cfg:
            self.pretrained_var.set(bool(cfg['pretrained']))
        messagebox.showinfo('Loaded', f'Config loaded from {p}')

    def mainloop(self):
        self.root.mainloop()


class TextHandler(logging.Handler):
    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record) + '\n'
        try:
            self.text_widget.configure(state='normal')
            self.text_widget.insert('end', msg)
            self.text_widget.see('end')
            self.text_widget.configure(state='disabled')
        except Exception:
            pass

# ---------------- CLI entrypoint ----------------

def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/final')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--models_dir', default='models')
    parser.add_argument('--freeze_epochs', type=int, default=3)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--use_sampler', action='store_true')
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--scheduler', choices=['reduce_on_plateau','cosine'], default='reduce_on_plateau')
    parser.add_argument('--early_stopping_patience', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--base_model', choices=['resnet50','resnet18'], default='resnet50')
    parser.add_argument('--tensorboard', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--gui', action='store_true')
    args = parser.parse_args()

    if args.gui:
        if not _HAS_TK:
            logger.error('Tkinter not available')
            return
        app = TrainerGUI()
        app.mainloop()
        return

    cfg = {
        'data': args.data,
        'models_dir': args.models_dir,
        'epochs': args.epochs,
        'batch': args.batch,
        'lr': args.lr,
        'freeze_epochs': args.freeze_epochs,
        'num_classes': args.num_classes,
        'image_size': args.image_size,
        'pretrained': args.pretrained,
        'use_sampler': args.use_sampler,
        'grad_clip': args.grad_clip,
        'scheduler': args.scheduler,
        'early_stopping_patience': args.early_stopping_patience,
        'resume': args.resume,
        'base_model': args.base_model,
        'tensorboard': args.tensorboard,
        'amp': args.amp
    }
    run_training(cfg)

if __name__ == '__main__':
    main_cli()
