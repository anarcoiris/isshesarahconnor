#!/usr/bin/env python3
"""
Diagnóstico y calibrado para classifier PyTorch (resnet).

Uso ejemplo:
  python scripts/diagnose_model.py --model models/best_model.pth --data data/final --split val --base_model resnet50 --image_size 224 --labels labels.json --calibrate --out calibrated.pth

Funciones:
 - evalúa métricas (accuracy, classification report)
 - muestra ROC AUC (binario), matriz de confusión
 - calcula ECE (expected calibration error)
 - aplica temperature-scaling (calibración) y guarda temperatura
 - exporta CSV con probabilidades por imagen (para inspección)
"""
from __future__ import annotations
import argparse
import os
import json
import time
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# -------- helpers ----------
def load_labels(path: Optional[str], default: List[str]) -> List[str]:
    if path and os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                # try order by int keys
                try:
                    return [v for k,v in sorted(data.items(), key=lambda kv: int(kv[0]))]
                except Exception:
                    return list(data.values())
        except Exception as e:
            print("Warning loading labels.json:", e)
    return default

def build_model(base: str, num_classes: int, pretrained: bool=False):
    if base == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    elif base == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError("base not supported")
    in_f = m.fc.in_features
    m.fc = nn.Linear(in_f, num_classes)
    return m

def flexible_load_state(model: torch.nn.Module, path: str):
    raw = torch.load(path, map_location='cpu')
    # if dict with model_state / state_dict
    if isinstance(raw, dict):
        for key in ("model_state", "state_dict", "model_state_dict"):
            if key in raw:
                sd = raw[key]
                break
        else:
            # maybe it's already a state_dict (keys look like params)
            sd = raw
    else:
        sd = raw
    # strip module. prefix if needed
    new_sd = {}
    changed = False
    for k,v in sd.items():
        if k.startswith("module."):
            new_sd[k[7:]] = v; changed = True
        else:
            new_sd[k] = v
    if changed:
        print("Stripped 'module.' prefix from state_dict keys.")
    model.load_state_dict(new_sd)
    return model

def get_transform(image_size:int):
    return transforms.Compose([
        transforms.Resize(int(image_size*1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

# ---------- calibration: temperature scaling ----------
class TemperatureScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

def fit_temperature(logits: torch.Tensor, labels: torch.Tensor, device: torch.device = torch.device('cpu')) -> float:
    """
    Fit a single temperature using NLL on validation logits.
    logits: tensor (N, C)
    labels: tensor (N,)
    returns fitted temperature (float)
    """
    logits = logits.to(device)
    labels = labels.to(device)
    temp = TemperatureScaler().to(device)
    optimizer = torch.optim.LBFGS([temp.temperature], lr=0.01, max_iter=200, line_search_fn='strong_wolfe')

    nll_criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = nll_criterion(temp(logits), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    t = float(temp.temperature.detach().cpu().numpy())
    return t

# ---------- utilities ----------
def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute ECE (basic).
    probs: predicted prob for predicted class (N,)
    labels: true labels (N,)
    """
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i+1]) if i < n_bins-1 else (probs >= bins[i]) & (probs <= bins[i+1])
        if mask.sum() == 0:
            continue
        acc = (labels[mask] == (probs[mask] >= 0.5)).mean() if len(labels[mask])>0 else 0.0
        conf = probs[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)

# ---------- main evaluation ----------
def evaluate_model(model, loader, device, tta:bool=False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    all_probs = []
    all_preds = []
    all_labels = []
    all_logits = []
    with torch.no_grad():
        for xb, y in loader:
            xb = xb.to(device)
            y = y.to(device)
            logits = model(xb)
            probs = F.softmax(logits, dim=1)
            all_logits.append(logits.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_preds.append(probs.argmax(dim=1).cpu().numpy())
            all_labels.append(y.cpu().numpy())
    all_logits = np.vstack(all_logits)
    all_probs = np.vstack(all_probs)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return all_logits, all_probs, all_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', default='data/final')
    parser.add_argument('--split', choices=['val','test'], default='val')
    parser.add_argument('--labels', default='labels.json')
    parser.add_argument('--base_model', default='resnet50', choices=['resnet50','resnet18'])
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--calibrate', action='store_true', help='fit temperature on split and print calibrated metrics')
    parser.add_argument('--out', default=None, help='save calibrated temperature to this path (json)')
    args = parser.parse_args()

    labels = load_labels(args.labels, default=['class0','class1'])
    num_classes = len(labels)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device:", device)

    # build model and load
    model = build_model(args.base_model, num_classes, pretrained=args.pretrained)
    flexible_load_state(model, args.model)
    model = model.to(device)

    # dataset
    split_dir = os.path.join(args.data, args.split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"{split_dir} not found")
    tf = get_transform(args.image_size)
    ds = datasets.ImageFolder(split_dir, transform=tf)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=(device.type=='cuda'))

    # evaluate
    logits, probs, labels_true = evaluate_model(model, loader, device)
    preds = probs.argmax(axis=1)
    acc = accuracy_score(labels_true, preds)
    print("Acc:", acc)
    print("Classification report:")
    print(classification_report(labels_true, preds, target_names=ds.classes, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(labels_true, preds))

    # calibration metrics
    # for binary: use prob of positive class index (we take argmax=1 prob)
    if num_classes == 2:
        pos_probs = probs[:,1]  # assuming class 1 is positive (adjust if needed)
        try:
            auc = roc_auc_score(labels_true, pos_probs)
            print("ROC AUC (pos class):", auc)
        except Exception:
            pass
        brier = brier_score_loss(labels_true, pos_probs)
        print("Brier score:", brier)

        ece = expected_calibration_error(pos_probs, labels_true, n_bins=15)
        print("ECE:", ece)

        # Reliability diagram
        frac_pos, mean_pred = calibration_curve(labels_true, pos_probs, n_bins=10)
        plt.figure(figsize=(6,6))
        plt.plot(mean_pred, frac_pos, marker='o', label='empirical')
        plt.plot([0,1],[0,1], linestyle='--', label='perfect')
        plt.xlabel('Predicted prob')
        plt.ylabel('Fraction positive')
        plt.title('Reliability diagram')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("reliability_diagram.png")
        print("Saved reliability_diagram.png")

    # histogram of max-probs
    max_probs = probs.max(axis=1)
    plt.figure(figsize=(6,3))
    plt.hist(max_probs, bins=30)
    plt.title("Histogram of max predicted probabilities")
    plt.tight_layout()
    plt.savefig("probs_hist.png")
    print("Saved probs_hist.png")

    # print top ambiguous examples (closest to 0.5 for binary)
    if num_classes==2:
        idx_amb = np.argsort(np.abs(pos_probs - 0.5))[:20]
        print("\nExamples most ambiguous (idx, true, prob_pos):")
        for i in idx_amb:
            print(i, labels_true[i], float(pos_probs[i]))

    # temperature scaling
    if args.calibrate and num_classes==2:
        print("Fitting temperature scaling on this split...")
        logits_t = torch.from_numpy(logits)
        labels_t = torch.from_numpy(labels_true).long()
        temp = fit_temperature(logits_t, labels_t, device=device)
        print("Fitted temperature:", temp)
        # apply temperature
        calibrated_probs = F.softmax(torch.from_numpy(logits) / temp, dim=1).numpy()
        pos_cal = calibrated_probs[:,1]
        cal_brier = brier_score_loss(labels_true, pos_cal)
        cal_ece = expected_calibration_error(pos_cal, labels_true, n_bins=15)
        try:
            cal_auc = roc_auc_score(labels_true, pos_cal)
        except Exception:
            cal_auc = None
        print("After calibration -> Brier:", cal_brier, "ECE:", cal_ece, "ROC AUC:", cal_auc)

        # save temperature
        if args.out:
            with open(args.out, 'w', encoding='utf-8') as f:
                json.dump({"temperature": float(temp)}, f, indent=2)
            print("Saved temperature to", args.out)
    # Save per-sample csv for inspection
    import csv
    out_csv = "preds_by_image.csv"
    root = Path(split_dir)
    # loader.dataset.imgs gives (path, idx)
    rows = []
    for (path, idx), prob_row in zip(loader.dataset.samples, probs):
        rows.append([path, idx] + prob_row.tolist())
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["path","label_idx"] + [f"p_{i}" for i in range(probs.shape[1])])
        writer.writerows(rows)
    print("Saved per-sample probabilities to", out_csv)

if __name__ == "__main__":
    main()
